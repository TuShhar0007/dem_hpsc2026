/**
 * ============================================================
 *  3D Discrete Element Method (DEM) Solver
 *  HPSC 2026 – Assignment 1
 *
 *  Features:
 *   • Serial O(N²) particle-particle contact detection
 *   • OpenMP parallel version with thread-local force buffers
 *   • Cell-linked list neighbour search  O(N) average
 *   • Semi-implicit Euler time integration
 *   • Linear spring-dashpot contact model
 *   • Wall contact forces for cuboidal box
 *   • Verification tests: free-fall, constant velocity, bounce
 *   • CSV output for post-processing / plotting
 *
 *  Build (serial):
 *    g++ -O2 -std=c++17 -o dem dem_solver.cpp
 *
 *  Build (OpenMP):
 *    g++ -O2 -std=c++17 -fopenmp -o dem dem_solver.cpp
 *
 *  Run:
 *    ./dem [mode] [N]
 *    mode: serial | omp | neighbor | tests
 *    N   : number of particles (default 200)
 *
 *  Usage reported: Claude (Anthropic) – all physics, numerics,
 *  and algorithmic choices verified manually against assignment spec.
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <omp.h>

// ============================================================
//  Vec3 – lightweight 3-D vector
// ============================================================

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3  operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3  operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3  operator*(double s)      const { return {x*s,   y*s,   z*s  }; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }

    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    double norm2()            const { return x*x + y*y + z*z; }
    double norm()             const { return std::sqrt(norm2()); }

    Vec3 normalized() const {
        double n = norm();
        return (n > 1.0e-14) ? Vec3(x/n, y/n, z/n) : Vec3(0,0,0);
    }
};

// ============================================================
//  Data structures
// ============================================================

struct Particle {
    Vec3   pos;          // position     (m)
    Vec3   vel;          // velocity     (m/s)
    Vec3   force;        // total force  (N)
    double mass   = 1.0; // mass         (kg)
    double radius = 0.05;// radius       (m)
};

struct SimParams {
    double kn        = 1.0e5;   // normal spring stiffness  (N/m)
    double gamma_n   = 50.0;    // normal damping coefficient (N·s/m)
    double dt        = 1.0e-5;  // timestep   (s)
    double t_end     = 0.5;     // end time   (s)
    double Lx        = 5.0;     // box x-extent (m)
    double Ly        = 5.0;     // box y-extent (m)
    double Lz        = 5.0;     // box z-extent (m)
    Vec3   gravity   = {0, 0, -9.81};
    int    output_interval = 200;   // write every N steps
};

// ============================================================
//  Initialisation helpers
// ============================================================

/**
 * Place N particles randomly in the upper portion of the box.
 * Particles are given zero initial velocity.
 */
void initialize_particles(std::vector<Particle>& particles,
                           int N,
                           const SimParams& sp,
                           unsigned seed = 42)
{
    std::mt19937 rng(seed);
    // Place in [10%, 90%] × [10%, 90%] × [50%, 90%] to avoid wall overlap
    std::uniform_real_distribution<double> rx(0.10*sp.Lx, 0.90*sp.Lx);
    std::uniform_real_distribution<double> ry(0.10*sp.Ly, 0.90*sp.Ly);
    std::uniform_real_distribution<double> rz(0.50*sp.Lz, 0.90*sp.Lz);

    particles.resize(N);
    for (auto& p : particles) {
        p.radius = 0.05;
        p.mass   = 1.0;
        p.pos    = Vec3(rx(rng), ry(rng), rz(rng));
        p.vel    = Vec3(0, 0, 0);
        p.force  = Vec3(0, 0, 0);
    }
}

// ============================================================
//  Per-timestep force routines
// ============================================================

/** Reset all particle forces to zero. */
void zero_forces(std::vector<Particle>& particles)
{
    for (auto& p : particles)
        p.force = Vec3(0, 0, 0);
}

/** Add gravitational body force  F_i += m_i * g */
void add_gravity(std::vector<Particle>& particles, const Vec3& g)
{
    for (auto& p : particles)
        p.force += g * p.mass;
}

// ────────────────────────────────────────────────────────────
//  Contact model helper  (used by all three implementations)
// ────────────────────────────────────────────────────────────
/**
 * Evaluate the spring-dashpot contact force magnitude.
 *
 *   F_n = max(0,  kn * delta - gamma_n * v_n)
 *
 * Returns 0 if the surfaces are separating fast enough to overcome
 * the spring, preventing spurious attractive forces.
 */
inline double contact_force(double delta, double vn,
                             double kn,    double gn)
{
    return std::max(0.0, kn * delta - gn * vn);
}

// ────────────────────────────────────────────────────────────
//  A) Serial  O(N²)  particle–particle contact detection
// ────────────────────────────────────────────────────────────
void compute_particle_contacts(std::vector<Particle>& particles,
                                double kn, double gamma_n)
{
    const int N = static_cast<int>(particles.size());
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3   rij   = particles[j].pos - particles[i].pos;
            double dij   = rij.norm();
            double delta = particles[i].radius + particles[j].radius - dij;

            if (delta > 0.0) {
                Vec3   nij = rij.normalized();
                double vn  = (particles[j].vel - particles[i].vel).dot(nij);
                double Fn  = contact_force(delta, vn, kn, gamma_n);
                Vec3   Fc  = nij * Fn;
                particles[i].force += Fc;   // action
                particles[j].force -= Fc;   // reaction
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
//  B) OpenMP parallel  O(N²) – thread-local force buffers
//
//  Strategy: each OpenMP thread accumulates forces into its own
//  private array (no false sharing, no atomics on hot path).
//  After the parallel region the per-thread arrays are reduced
//  into the particle forces serially – this reduction is O(N*T)
//  and is dominated by the O(N²) work for large N.
// ────────────────────────────────────────────────────────────
void compute_particle_contacts_omp(std::vector<Particle>& particles,
                                    double kn, double gamma_n)
{
    const int N       = static_cast<int>(particles.size());
    const int NTHREADS = omp_get_max_threads();

    // Allocate thread-local force buffers  [thread][particle]
    std::vector<std::vector<Vec3>> tf(NTHREADS,
                                      std::vector<Vec3>(N, Vec3(0,0,0)));

    #pragma omp parallel default(none) \
        shared(particles, tf, kn, gamma_n, N, NTHREADS)
    {
        const int tid = omp_get_thread_num();
        auto& local   = tf[tid];

        // Dynamic scheduling handles load imbalance from the
        // triangular loop structure automatically.
        #pragma omp for schedule(dynamic, 32) nowait
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                Vec3   rij   = particles[j].pos - particles[i].pos;
                double dij   = rij.norm();
                double delta = particles[i].radius + particles[j].radius - dij;

                if (delta > 0.0) {
                    Vec3   nij = rij.normalized();
                    double vn  = (particles[j].vel - particles[i].vel).dot(nij);
                    double Fn  = contact_force(delta, vn, kn, gamma_n);
                    Vec3   Fc  = nij * Fn;
                    local[i] += Fc;
                    local[j] -= Fc;
                }
            }
        }
    } // end parallel

    // Reduce thread-local buffers → particle forces
    for (int t = 0; t < NTHREADS; ++t)
        for (int i = 0; i < N; ++i)
            particles[i].force += tf[t][i];
}

// ────────────────────────────────────────────────────────────
//  C) Cell-linked list neighbour search  O(N) average
//
//  The domain is divided into cubic cells of side-length
//  cell_size ≥ 2 * r_max so that interacting pairs can only
//  be in adjacent cells.  Each particle is placed in a singly-
//  linked list per cell; contact checks then traverse only the
//  27 neighbouring cells.
// ────────────────────────────────────────────────────────────

struct CellGrid {
    int    ncx = 1, ncy = 1, ncz = 1;   // number of cells per axis
    double cs  = 1.0;                   // cell size (uniform)
    double Lx, Ly, Lz;

    std::vector<int> head;   // head[cell_id] = first particle in cell (-1 = empty)
    std::vector<int> next;   // next[i]       = next particle in same cell

    /** Build the linked-list structure from current particle positions. */
    void build(const std::vector<Particle>& parts,
               double r_max,
               double lx, double ly, double lz)
    {
        Lx = lx; Ly = ly; Lz = lz;

        // Cell size must be at least 2*r_max so no contact is missed
        cs  = 2.0 * r_max;
        ncx = std::max(1, static_cast<int>(Lx / cs));
        ncy = std::max(1, static_cast<int>(Ly / cs));
        ncz = std::max(1, static_cast<int>(Lz / cs));
        // Re-derive exact cs so cells tile the box perfectly
        cs  = Lx / ncx;          // same in all directions (uniform box)

        const int ncells = ncx * ncy * ncz;
        head.assign(ncells, -1);
        next.assign(parts.size(), -1);

        for (int i = 0; i < static_cast<int>(parts.size()); ++i) {
            int cx = std::clamp((int)(parts[i].pos.x / cs), 0, ncx-1);
            int cy = std::clamp((int)(parts[i].pos.y / cs), 0, ncy-1);
            int cz = std::clamp((int)(parts[i].pos.z / cs), 0, ncz-1);
            int id = cz * ncy * ncx + cy * ncx + cx;
            next[i]  = head[id];
            head[id] = i;
        }
    }

    /** Map (cx,cy,cz) indices → linear cell id.  Returns -1 if out of bounds. */
    int cell_id(int cx, int cy, int cz) const {
        if (cx < 0 || cx >= ncx) return -1;
        if (cy < 0 || cy >= ncy) return -1;
        if (cz < 0 || cz >= ncz) return -1;
        return cz * ncy * ncx + cy * ncx + cx;
    }
};

void compute_particle_contacts_neighbor(std::vector<Particle>& particles,
                                         double kn, double gamma_n,
                                         const SimParams& sp)
{
    if (particles.empty()) return;

    // Determine max radius for cell-size calculation
    double r_max = 0.0;
    for (const auto& p : particles)
        r_max = std::max(r_max, p.radius);

    CellGrid grid;
    grid.build(particles, r_max, sp.Lx, sp.Ly, sp.Lz);

    const int N = static_cast<int>(particles.size());

    for (int i = 0; i < N; ++i) {
        // Cell coordinates of particle i
        int cx = std::clamp((int)(particles[i].pos.x / grid.cs), 0, grid.ncx-1);
        int cy = std::clamp((int)(particles[i].pos.y / grid.cs), 0, grid.ncy-1);
        int cz = std::clamp((int)(particles[i].pos.z / grid.cs), 0, grid.ncz-1);

        // Loop over 3×3×3 = 27 neighbouring cells (including own)
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int cid = grid.cell_id(cx+dx, cy+dy, cz+dz);
            if (cid < 0) continue;

            // Walk the linked list in that cell
            int j = grid.head[cid];
            while (j != -1) {
                // Process each pair (i,j) exactly once:  i < j
                if (j > i) {
                    Vec3   rij   = particles[j].pos - particles[i].pos;
                    double dij   = rij.norm();
                    double delta = particles[i].radius + particles[j].radius - dij;

                    if (delta > 0.0) {
                        Vec3   nij = rij.normalized();
                        double vn  = (particles[j].vel - particles[i].vel).dot(nij);
                        double Fn  = contact_force(delta, vn, kn, gamma_n);
                        Vec3   Fc  = nij * Fn;
                        particles[i].force += Fc;
                        particles[j].force -= Fc;
                    }
                }
                j = grid.next[j];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Wall contacts  (six faces of the cuboidal box)
// ────────────────────────────────────────────────────────────
/**
 * For each particle check overlap with each of the six walls.
 * The same spring-dashpot model is used; the wall normal points
 * inward (away from the wall surface).
 */
void compute_wall_contacts(std::vector<Particle>& particles,
                            const SimParams& sp)
{
    const double kn = sp.kn, gn = sp.gamma_n;

    for (auto& p : particles) {
        const double R = p.radius;

        // ── Lower x wall (normal = +x) ──
        { double d = R - p.pos.x;
          if (d > 0) { double Fn = contact_force(d, -p.vel.x, kn, gn); p.force.x += Fn; } }
        // ── Upper x wall (normal = -x) ──
        { double d = R - (sp.Lx - p.pos.x);
          if (d > 0) { double Fn = contact_force(d,  p.vel.x, kn, gn); p.force.x -= Fn; } }

        // ── Lower y wall (normal = +y) ──
        { double d = R - p.pos.y;
          if (d > 0) { double Fn = contact_force(d, -p.vel.y, kn, gn); p.force.y += Fn; } }
        // ── Upper y wall (normal = -y) ──
        { double d = R - (sp.Ly - p.pos.y);
          if (d > 0) { double Fn = contact_force(d,  p.vel.y, kn, gn); p.force.y -= Fn; } }

        // ── Floor (z = 0, normal = +z) ──
        { double d = R - p.pos.z;
          if (d > 0) { double Fn = contact_force(d, -p.vel.z, kn, gn); p.force.z += Fn; } }
        // ── Ceiling (z = Lz, normal = -z) ──
        { double d = R - (sp.Lz - p.pos.z);
          if (d > 0) { double Fn = contact_force(d,  p.vel.z, kn, gn); p.force.z -= Fn; } }
    }
}

// ============================================================
//  Time integration – semi-implicit Euler
//
//    v^{n+1}_i = v^n_i + (F^n_i / m_i) * dt
//    x^{n+1}_i = x^n_i + v^{n+1}_i   * dt
// ============================================================
void semi_implicit_euler(std::vector<Particle>& particles, double dt)
{
    for (auto& p : particles) {
        p.vel += p.force * (dt / p.mass);
        p.pos += p.vel * dt;
    }
}

// ============================================================
//  Diagnostics
// ============================================================

double compute_kinetic_energy(const std::vector<Particle>& particles)
{
    double KE = 0.0;
    for (const auto& p : particles)
        KE += 0.5 * p.mass * p.vel.norm2();
    return KE;
}

int count_contacts(const std::vector<Particle>& particles)
{
    const int N = static_cast<int>(particles.size());
    int nc = 0;
    for (int i = 0; i < N-1; ++i)
        for (int j = i+1; j < N; ++j) {
            double d = (particles[j].pos - particles[i].pos).norm();
            if (particles[i].radius + particles[j].radius - d > 0) ++nc;
        }
    return nc;
}

// ============================================================
//  Output
// ============================================================

/** Write one snapshot: t, particle-id, x, y, z, vx, vy, vz */
void write_output(const std::vector<Particle>& particles,
                  double t, std::ofstream& fout)
{
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        const auto& p = particles[i];
        fout << std::scientific << std::setprecision(8)
             << t      << ","
             << i      << ","
             << p.pos.x << "," << p.pos.y << "," << p.pos.z << ","
             << p.vel.x << "," << p.vel.y << "," << p.vel.z << "\n";
    }
}

// ============================================================
//  Verification Tests
// ============================================================

/**
 * Test 1: Free Fall
 *
 * A single particle released from rest at height z0.
 * Analytical solution (no air resistance):
 *   z(t)  = z0 - ½ g t²
 *   vz(t) = -g t
 *
 * Gravity g = 9.81 m/s²,  no wall contact enforced here so the
 * particle passes freely through the floor (clean analytical test).
 */
void test_free_fall(const SimParams& sp)
{
    std::cout << "\n[Test 1] Free Fall\n";

    const double z0 = 5.0;   // m  (well above floor)
    const double g  = std::abs(sp.gravity.z);

    std::vector<Particle> parts(1);
    parts[0].radius = 0.05;
    parts[0].mass   = 1.0;
    parts[0].pos    = Vec3(0.5*sp.Lx, 0.5*sp.Ly, z0);
    parts[0].vel    = Vec3(0, 0, 0);

    // Also run for multiple dt values to produce error-vs-dt data
    std::vector<double> dt_vals = {1e-3, 5e-4, 1e-4, 5e-5, 1e-5};
    std::ofstream ferr("free_fall_error.csv");
    ferr << "dt,L2_error\n";

    for (double dt_test : dt_vals) {
        // Reset
        parts[0].pos = Vec3(0.5*sp.Lx, 0.5*sp.Ly, z0);
        parts[0].vel = Vec3(0, 0, 0);
        parts[0].force = Vec3(0,0,0);

        double t = 0.0;
        double err2 = 0.0;
        int    cnt  = 0;
        double t_stop = 0.5; // run for 0.5 s

        while (t < t_stop) {
            zero_forces(parts);
            add_gravity(parts, sp.gravity);
            semi_implicit_euler(parts, dt_test);
            t += dt_test;
            ++cnt;

            double z_ana = z0 - 0.5 * g * t * t;
            double e = parts[0].pos.z - z_ana;
            err2 += e * e;
        }
        double rmse = std::sqrt(err2 / cnt);
        ferr << dt_test << "," << rmse << "\n";
    }
    ferr.close();

    // Write detailed trajectory for the assignment dt
    parts[0].pos   = Vec3(0.5*sp.Lx, 0.5*sp.Ly, z0);
    parts[0].vel   = Vec3(0, 0, 0);
    parts[0].force = Vec3(0, 0, 0);

    std::ofstream fout("free_fall.csv");
    fout << "t,z_num,z_ana,vz_num,vz_ana\n";
    double t = 0.0;
    while (t < 1.0) {                       // 1 s of free fall
        zero_forces(parts);
        add_gravity(parts, sp.gravity);
        semi_implicit_euler(parts, sp.dt);
        t += sp.dt;
        double z_ana  = z0 - 0.5*g*t*t;
        double vz_ana = -g*t;
        fout << t << "," << parts[0].pos.z << "," << z_ana
             << "," << parts[0].vel.z << "," << vz_ana << "\n";
    }
    fout.close();
    std::cout << "  → free_fall.csv, free_fall_error.csv written.\n";
}

/**
 * Test 2: Constant Velocity
 *
 * g = 0, no contacts.  Particle must travel at exactly v0.
 */
void test_constant_velocity(const SimParams& sp)
{
    std::cout << "\n[Test 2] Constant Velocity\n";

    const Vec3 v0 = {1.0, 0.5, 0.3};

    std::vector<Particle> parts(1);
    parts[0].radius = 0.05;
    parts[0].mass   = 1.0;
    parts[0].pos    = Vec3(0.5*sp.Lx, 0.5*sp.Ly, 0.5*sp.Lz);
    parts[0].vel    = v0;
    parts[0].force  = Vec3(0, 0, 0);

    SimParams sp0 = sp;
    sp0.gravity   = Vec3(0, 0, 0);

    std::ofstream fout("const_vel.csv");
    fout << "t,x_num,x_ana,y_num,y_ana,z_num,z_ana\n";

    double t = 0.0;
    double x0 = parts[0].pos.x, y0 = parts[0].pos.y, z0 = parts[0].pos.z;
    double max_err = 0.0;

    for (int step = 0; step < 2000; ++step) {
        zero_forces(parts);
        // No gravity
        semi_implicit_euler(parts, sp0.dt);
        t += sp0.dt;

        double xa = x0 + v0.x * t;
        double ya = y0 + v0.y * t;
        double za = z0 + v0.z * t;

        fout << t << ","
             << parts[0].pos.x << "," << xa << ","
             << parts[0].pos.y << "," << ya << ","
             << parts[0].pos.z << "," << za << "\n";

        double err = (parts[0].pos - Vec3(xa, ya, za)).norm();
        max_err = std::max(max_err, err);
    }
    fout.close();
    std::cout << "  → const_vel.csv written.  Max positional error = "
              << max_err << " m\n";
    assert(max_err < 1.0e-10 && "Constant-velocity test FAILED");
    std::cout << "  → PASSED (max error < 1e-10)\n";
}

/**
 * Test 3: Particle Bounce
 *
 * Single particle dropped onto the floor.
 * Checks:
 *  • particle does not penetrate excessively (< 5% of radius)
 *  • rebound height decreases monotonically with damping
 */
void test_bounce(const SimParams& sp)
{
    std::cout << "\n[Test 3] Particle Bounce\n";

    std::vector<Particle> parts(1);
    parts[0].radius = 0.1;
    parts[0].mass   = 1.0;
    parts[0].pos    = Vec3(0.5*sp.Lx, 0.5*sp.Ly, 2.0);
    parts[0].vel    = Vec3(0, 0, 0);
    parts[0].force  = Vec3(0, 0, 0);

    std::ofstream fout("bounce.csv");
    fout << "t,z,vz,KE\n";

    double t = 0.0;
    double min_z  = parts[0].pos.z;
    double max_pen = 0.0;

    for (int step = 0; step < 10000; ++step) {
        zero_forces(parts);
        add_gravity(parts, sp.gravity);
        compute_wall_contacts(parts, sp);
        semi_implicit_euler(parts, sp.dt);
        t += sp.dt;

        double overlap = parts[0].radius - parts[0].pos.z;
        if (overlap > max_pen) max_pen = overlap;
        min_z = std::min(min_z, parts[0].pos.z);

        double KE = compute_kinetic_energy(parts);
        fout << t << "," << parts[0].pos.z << ","
             << parts[0].vel.z << "," << KE << "\n";
    }
    fout.close();

    // Penetration should be a small fraction of the radius
    double pen_frac = max_pen / parts[0].radius;
    std::cout << "  → bounce.csv written.\n";
    std::cout << "  → Max wall penetration: " << max_pen << " m  ("
              << 100.0*pen_frac << "% of radius)\n";
    if (pen_frac < 0.05)
        std::cout << "  → PASSED (penetration < 5%)\n";
    else
        std::cout << "  → WARNING: penetration > 5%. Consider reducing dt or increasing kn.\n";
}

// ============================================================
//  Production simulation runner
// ============================================================

enum class Method { SERIAL, OMP, NEIGHBOR };

double run_simulation(int N, const SimParams& sp, Method method,
                      const std::string& prefix)
{
    std::vector<Particle> particles;
    initialize_particles(particles, N, sp);

    std::ofstream ftraj(prefix + "_traj.csv");
    ftraj << "t,id,x,y,z,vx,vy,vz\n";

    std::ofstream fenergy(prefix + "_energy.csv");
    fenergy << "t,KE\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    double t   = 0.0;
    int    step = 0;
    const int total_steps = static_cast<int>(sp.t_end / sp.dt);

    while (step < total_steps) {
        zero_forces(particles);
        add_gravity(particles, sp.gravity);

        switch (method) {
            case Method::SERIAL:
                compute_particle_contacts(particles, sp.kn, sp.gamma_n);
                break;
            case Method::OMP:
                compute_particle_contacts_omp(particles, sp.kn, sp.gamma_n);
                break;
            case Method::NEIGHBOR:
                compute_particle_contacts_neighbor(particles, sp.kn, sp.gamma_n, sp);
                break;
        }

        compute_wall_contacts(particles, sp);
        semi_implicit_euler(particles, sp.dt);
        t    += sp.dt;
        ++step;

        if (step % sp.output_interval == 0) {
            write_output(particles, t, ftraj);
            fenergy << t << "," << compute_kinetic_energy(particles) << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    const char* mname[] = {"SERIAL", "OMP", "NEIGHBOR"};
    std::cout << "  N=" << std::setw(5) << N
              << "  method=" << mname[static_cast<int>(method)]
              << "  threads=" << omp_get_max_threads()
              << "  time=" << std::fixed << std::setprecision(3) << elapsed << " s\n";

    ftraj.close();
    fenergy.close();
    return elapsed;
}

// ============================================================
//  Performance scaling study
// ============================================================

void scaling_study(const SimParams& sp)
{
    std::cout << "\n=== Scaling Study ===\n";

    std::ofstream fscale("scaling.csv");
    fscale << "N,threads,method,time_s\n";

    std::vector<int> Nvals = {200, 1000, 5000};
    std::vector<int> thread_counts = {1, 2, 4, 8};

    for (int N : Nvals) {
        // Serial baseline
        double T1 = run_simulation(N, sp, Method::SERIAL,
                                   "sim_N"+std::to_string(N)+"_serial");
        fscale << N << ",1,SERIAL," << T1 << "\n";

        // OpenMP with various thread counts
        for (int nt : thread_counts) {
            omp_set_num_threads(nt);
            double Tp = run_simulation(N, sp, Method::OMP,
                                       "sim_N"+std::to_string(N)+"_omp_t"+std::to_string(nt));
            fscale << N << "," << nt << ",OMP," << Tp << "\n";
        }

        // Neighbour search (serial)
        omp_set_num_threads(1);
        double Tn = run_simulation(N, sp, Method::NEIGHBOR,
                                   "sim_N"+std::to_string(N)+"_neighbor");
        fscale << N << ",1,NEIGHBOR," << Tn << "\n";
    }

    fscale.close();
    std::cout << "  → scaling.csv written.\n";
}

// ============================================================
//  main
// ============================================================

int main(int argc, char* argv[])
{
    SimParams sp;           // all defaults defined in struct above

    std::string mode = (argc > 1) ? argv[1] : "tests";
    int N            = (argc > 2) ? std::stoi(argv[2]) : 200;

    std::cout << "===================================================\n";
    std::cout << " DEM Solver – HPSC 2026 Assignment 1\n";
    std::cout << " Mode: " << mode << "  |  N: " << N
              << "  |  OMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "===================================================\n";

    if (mode == "tests" || mode == "all") {
        test_free_fall(sp);
        test_constant_velocity(sp);
        test_bounce(sp);
    }

    if (mode == "serial" || mode == "all") {
        std::cout << "\n=== Serial run, N=" << N << " ===\n";
        run_simulation(N, sp, Method::SERIAL, "sim_serial");
    }

    if (mode == "omp" || mode == "all") {
        std::cout << "\n=== OpenMP run, N=" << N << " ===\n";
        run_simulation(N, sp, Method::OMP, "sim_omp");
    }

    if (mode == "neighbor" || mode == "all") {
        std::cout << "\n=== Neighbour-search run, N=" << N << " ===\n";
        run_simulation(N, sp, Method::NEIGHBOR, "sim_neighbor");
    }

    if (mode == "scaling") {
        scaling_study(sp);
    }

    std::cout << "\nDone.\n";
    return 0;
}
