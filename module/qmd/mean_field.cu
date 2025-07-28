
#include "mean_field.cuh"
#include "shuffle.cuh"
#include "collision.cuh"

#include "nucleus.cuh"

#include <stdio.h>


namespace RT2QMD {
    namespace MeanField {


        __device__ void setDimension(int n) {
            assert(n >= 0 && n <= Buffer::MAX_DIMENSION_PARTICIPANT);

            int iter;
            Buffer::model_cached->current_field_size   = n;
            Buffer::model_cached->current_field_size_2 = n * n;

            // two body iteration
            iter = (int)ceilf((float)(n * n) / (float)blockDim.x);
            Buffer::model_cached->iter_2body = iter;

            // one body iteration
            iter = (int)ceilf((float)n / (float)blockDim.x);
            Buffer::model_cached->iter_1body = iter;

            // graduate iteration
            iter = (int)ceilf((float)(Buffer::Participant::SOA_MEMBERS * CUDA_WARP_SIZE) / (float)blockDim.x);
            Buffer::model_cached->iter_gphase = iter;
            iter = (int)ceilf((float)n / (float)(blockDim.x / CUDA_WARP_SIZE));
            Buffer::model_cached->iter_grh3d  = iter;
            iter = (int)ceilf((float)n / (float)CUDA_WARP_SIZE);
            Buffer::model_cached->iter_gwarp  = iter;
            iter = (int)ceilf((float)n / (float)CUDA_WARP_SIZE);
            Buffer::model_cached->iter_gblock = iter;
            iter = CUDA_WARP_SIZE * CUDA_WARP_SIZE / blockDim.x;
            Buffer::model_cached->iter_glane  = iter;
        }


        __device__ void cal2BodyQuantities(bool prepare_collision) {

            float  temp;
            float3 pi, pj;
            float  rij;
            float  rh1;

            Collision::CollisionSharedMem* smem = (Collision::CollisionSharedMem*)mcutil::cache_univ;

            if (prepare_collision && !threadIdx.x)
                smem->n_binary_candidate = 0;
            __syncthreads();

            for (int it = 0; it < Buffer::model_cached->iter_2body; ++it) {
                // indexing matrix
                int mai = it * blockDim.x + threadIdx.x;
                if (mai < Buffer::model_cached->current_field_size_2) {
                    // indexing partcipant & matrix, set memory offset
                    int i = mai / Buffer::model_cached->current_field_size;
                    int j = mai - i * Buffer::model_cached->current_field_size;
                    i  = (int)Buffer::model_cached->participant_idx[i] + Buffer::model_cached->offset_1d;
                    j  = (int)Buffer::model_cached->participant_idx[j] + Buffer::model_cached->offset_1d;
                    mai = mai + Buffer::model_cached->offset_2d;
                    bool is_binary_candidate = true;

                    // flag
                    int fi = Buffer::Participant::flags[i];
                    int fj = Buffer::Participant::flags[j];

                    // charge
                    bool ci = fi & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    bool cj = fj & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

                    // momentum
                    pi.x = Buffer::Participant::momentum_x[i];
                    pj.x = Buffer::Participant::momentum_x[j];
                    pi.y = Buffer::Participant::momentum_y[i];
                    pj.y = Buffer::Participant::momentum_y[j];
                    pi.z = Buffer::Participant::momentum_z[i];
                    pj.z = Buffer::Participant::momentum_z[j];

                    assertNAN(pi.x);
                    assertNAN(pj.x);
                    assertNAN(pi.y);
                    assertNAN(pj.y);
                    assertNAN(pi.z);
                    assertNAN(pj.z);

                    // energy
                    float mi = Buffer::Participant::mass[i];
                    float mj = Buffer::Participant::mass[j];

                    assert(mi > 0.f);
                    assert(mj > 0.f);

                    float ei = sqrtf(mi * mi + pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
                    float ej = sqrtf(mj * mj + pj.x * pj.x + pj.y * pj.y + pj.z * pj.z);

                    float rbrb = 0.f;
                    float rij2 = 0.f;
                    float pij2 = 0.f;
                    float gij2 = 0.f;

                    float pidr = 0.f;
                    float pjdr = 0.f;

                    // vector x
                    rij   = Buffer::Participant::position_x[i];
                    rij  -= Buffer::Participant::position_x[j];  // dr.x
                    assertNAN(rij);
                    pidr += rij * pi.x;
                    pjdr += rij * pj.x;
                    temp  = pi.x + pj.x;   // p4ij.x
                    rbrb += rij  * temp;   // rij.x  * p4ij.x
                    gij2 += temp * temp;   // p4ij.x * p4ij.x
                    rij2 += rij  * rij;    // rij.x  * rij.x  &&  rsq
                    temp  = pi.x - pj.x;   // pij.x
                    pij2 += temp * temp;   // pij.x  * pij.x

                    // vector y
                    rij   = Buffer::Participant::position_y[i];
                    rij  -= Buffer::Participant::position_y[j];
                    assertNAN(rij);
                    pidr += rij * pi.y;
                    pjdr += rij * pj.y;
                    temp  = pi.y + pj.y;
                    rbrb += rij * temp;
                    gij2 += temp * temp;
                    rij2 += rij * rij;
                    temp  = pi.y - pj.y;
                    pij2 += temp * temp;

                    // distance vector z
                    rij   = Buffer::Participant::position_z[i];
                    rij  -= Buffer::Participant::position_z[j];
                    assertNAN(rij);
                    pidr += rij * pi.z;
                    pjdr += rij * pj.z;
                    temp  = pi.z + pj.z;
                    rbrb += rij  * temp;
                    gij2 += temp * temp;
                    rij2 += rij  * rij;
                    temp  = pi.z - pj.z;
                    pij2 += temp * temp;

                    if (prepare_collision) {

                        // CM mass cutoff for binary
                        temp = ei + ej;
                        float e2cm = temp * temp - gij2;
                        temp = mi + mj + 0.02;
                        if (e2cm < temp * temp)
                            is_binary_candidate = false;

                        // impact parameter
                        float pij  = pi.x * pj.x + pi.y * pj.y + pi.z * pj.z;
                        temp = mi * mj / pij;
                        float aij  = 1.f - temp * temp;
                        float bij  = pidr / mi - pjdr * mj / pij;
                        temp = pidr / mi;
                        float cij  = rij2 + temp * temp;
                        float brel = sqrtf(fabsf(cij - bij * bij / aij));

                        if (brel > constants::FBCMAX0)
                            is_binary_candidate = false;

                        // minimum distance-of approach time
                        temp = -pjdr / mj + pidr * mj / pij;
                        float ti = ( pidr / mi - bij  / aij) * ei / mi;
                        float tj = (-pjdr / mj - temp / aij) * ej / mj;
                        if (fabsf(ti + tj) > constants::PROPAGATE_DT)
                            is_binary_candidate = false;
                    }

                    temp  = ei + ej;
                    rbrb /= temp;         // rij * p4ij.boost = (rij.x * p4ij.x + rij.y * p4ij.y + rij.z * p4ij.z) / p4ij.e
                    gij2 /= temp * temp;  // p4ij.boost.mag2
                    gij2 = 1.f / sqrtf(1.f - gij2); // gammaij
                    gij2  = gij2 * gij2;
                    rij2 += gij2 * rbrb * rbrb;

                    // mean field matrix element

                    if (prepare_collision) {

                        // flags
                        if (rij2 > constants::FDELTAR2) 
                            is_binary_candidate = false;
                        if ((fi & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE) &&
                            (fj & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE))
                            is_binary_candidate = false;
                        if ((fi & PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET) &&
                            (fj & PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET))
                            is_binary_candidate = false;

                        if (i >= j)
                            is_binary_candidate = false;

                        int bc_idx;
                        if (is_binary_candidate) {
                            bc_idx = atomicAdd(&smem->n_binary_candidate, 1);
                            if (bc_idx < Collision::BINARY_CANDIDATE_MAX) {
                                smem->binary_candidate[bc_idx].y = (unsigned char)(i - Buffer::model_cached->offset_1d);
                                smem->binary_candidate[bc_idx].x = (unsigned char)(j - Buffer::model_cached->offset_1d);
                            }
                        }
                    }
                    
                    // distance terms
                    Buffer::MeanField::rr2[mai]  = rij2;
                    Buffer::MeanField::rbij[mai] = gij2 * rbrb;
                    temp  = ci == cj ? 0.f : constants::MASS_DIF_2 / (ei + ej);
                    temp  = temp * temp;
                    Buffer::MeanField::pp2[mai]  = pij2 + gij2 * temp - (ei - ej) * (ei - ej);

                    // Gauss term
                    temp  = -rij2 * constants::D_SKYRME_C0;
                    rh1   = temp > constants::PAULI_EPSX ? expf(temp) : 0.f;
                    rh1   = i == j ? 0.f : rh1;
                    Buffer::MeanField::rha[mai]  = rh1;

                    // Coulomb term
                    rij2 += constants::COULOMB_EPSILON;
                    temp  = sqrtf(rij2);  // rrs
                    temp  = temp * constants::D_SKYRME_C0_S < 5.8
                        ? erff(temp * constants::D_SKYRME_C0_S) / temp
                        : 1.f / temp;  // erfij
                    temp  = i == j ? 0.f : temp;
                    Buffer::MeanField::rhe[mai] = (float)(ci * cj) * temp;
                    Buffer::MeanField::rhc[mai] = (float)(ci * cj) * (-temp + constants::D_COULOMB * rh1) / rij2;
                }
                __syncthreads();
            }

            __syncthreads();
        }


        __device__ void calGraduate(bool post_step) {
            int warp_idx = threadIdx.x %  CUDA_WARP_SIZE;
            int lane_idx = threadIdx.x >> CUDA_WARP_SHIFT;
            int lane_dim = blockDim.x  >> CUDA_WARP_SHIFT;
            int mi;  // matrix index (1d)

            // rho3 initialization
            float rho3;
            for (int ib = 0; ib < Buffer::model_cached->iter_grh3d; ++ib) {
                int i = lane_idx + ib * lane_dim;
                rho3  = 0.f;
                for (int jw = 0; jw < Buffer::model_cached->iter_gwarp; ++jw) {
                    int j = warp_idx + jw * CUDA_WARP_SIZE;
                    if (i < Buffer::model_cached->current_field_size && 
                        j < Buffer::model_cached->current_field_size) {
                        mi = i * Buffer::model_cached->current_field_size + j
                            + Buffer::model_cached->offset_2d;
                        rho3 += Buffer::MeanField::rha[mi];
                    }
                    __syncthreads();
                }
                rho3 = rsum32_(rho3);
                rho3 = powf(rho3, constants::G_SKYRME_GAMM);
                if (i < Buffer::model_cached->current_field_size)
                    Buffer::MeanField::rh3d[i + Buffer::model_cached->offset_1d] = rho3;
                __syncthreads();
            }

            // gradient
            int    pid;
            bool   ci, cj;
            float  temp;
            float  p_zero;
            float  mi_r;
            float3 ffr, ffp;  // rp gradient
            float3 betaij;
            float  ej;
            float  ccpp, grbb, ccrr, cij;
            // shared memory
            GradientSharedMem* smem = (GradientSharedMem*)mcutil::cache_univ;

            for (int ib = 0; ib < Buffer::model_cached->iter_gblock; ++ib) {
                // load global phase space for participant i
                for (int ip = 0; ip < Buffer::model_cached->iter_gphase; ++ip) {
                    int phid = ip * lane_dim + lane_idx;
                    int i    = ib * CUDA_WARP_SIZE + warp_idx;
                        pid  = (int)Buffer::model_cached->participant_idx[i] + Buffer::model_cached->offset_1d;
                    if (phid < Buffer::Participant::SOA_MEMBERS && i < Buffer::model_cached->current_field_size) {
                        mcutil::cache_univ[NUCLEI_SHARED_MEM_OFFSET + CUDA_WARP_SIZE * phid + warp_idx]
                            = Buffer::Participant::ps[phid][pid];
                    }
                    __syncthreads();
                }

                pid = ib * CUDA_WARP_SIZE + warp_idx;
                if (pid < Buffer::model_cached->current_field_size) {
                    // set energy
                    //temp = smem->flag[warp_idx] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON
                    //    ? constants::MASS_PROTON
                    //    : constants::MASS_NEUTRON;
                    temp  = smem->mass[warp_idx];
                    temp  = temp * temp;
                    temp += smem->px[warp_idx] * smem->px[warp_idx];
                    temp += smem->py[warp_idx] * smem->py[warp_idx];
                    temp += smem->pz[warp_idx] * smem->pz[warp_idx];
                    smem->ei[warp_idx] = sqrtf(temp);

                    // rh3d
                    pid += Buffer::model_cached->offset_1d;
                    smem->rh3d[warp_idx] = Buffer::MeanField::rh3d[pid];
                }

                // R-JQMD
                for (int iw = 0; iw < Buffer::model_cached->iter_glane; ++iw) {
                    int  phid = iw * lane_dim + lane_idx;
                    int  i    = ib * CUDA_WARP_SIZE + phid;
                    bool val  = i < Buffer::model_cached->current_field_size;
                    i  = val ? i : Buffer::model_cached->current_field_size - 1;
                    ci = smem->flag[phid] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    smem->vi[phid] = getPotential(i, ci);
                    __syncthreads();
                }

                for (int iw = 0; iw < Buffer::model_cached->iter_glane; ++iw) {
                    int phid = iw * lane_dim + lane_idx;
                    int i    = ib * CUDA_WARP_SIZE + phid;
                    ffr.x = 0.f;
                    ffr.y = 0.f;
                    ffr.z = 0.f;
                    ffp.x = 0.f;
                    ffp.y = 0.f;
                    ffp.z = 0.f;
                    if (i < Buffer::model_cached->current_field_size) {
                        ci     = smem->flag[phid] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

                        // R-JQMD
                        mi_r   = smem->mass[phid];
                        p_zero = sqrtf(smem->ei[phid] * smem->ei[phid] + 2.f * mi_r * smem->vi[phid]);
                        mi_r  /= p_zero;

                        if (!warp_idx) {
                            ffr.x = smem->px[phid] / p_zero;
                            ffr.y = smem->py[phid] / p_zero;
                            ffr.z = smem->pz[phid] / p_zero;
                        }
                    }
                    __syncthreads();
                    for (int jw = 0; jw < Buffer::model_cached->iter_gwarp; ++jw) {
                        int j = warp_idx + jw * CUDA_WARP_SIZE;
                        if (i < Buffer::model_cached->current_field_size &&
                            j < Buffer::model_cached->current_field_size) {
                            mi  = i * Buffer::model_cached->current_field_size + j
                                   + Buffer::model_cached->offset_2d;
                            pid = (int)Buffer::model_cached->participant_idx[j] + Buffer::model_cached->offset_1d;
                            cj  = Buffer::Participant::flags[pid] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                            betaij.x = Buffer::Participant::momentum_x[pid];
                            betaij.y = Buffer::Participant::momentum_y[pid];
                            betaij.z = Buffer::Participant::momentum_z[pid];
                            //ej = cj ? constants::MASS_PROTON : constants::MASS_NEUTRON;
                            ej  = Buffer::Participant::mass[pid];
                            ej  = ej * ej;
                            ej += betaij.x * betaij.x;
                            ej += betaij.y * betaij.y;
                            ej += betaij.z * betaij.z;
                            ej  = sqrtf(ej);  // relativistic energy of participant j

                            temp  = Buffer::MeanField::rha[mi];
                            ccpp  = 
                                constants::G_SKYRME_C0 + 
                                constants::G_SKYRME_C3 * (Buffer::MeanField::rh3d[pid] + smem->rh3d[phid]) + 
                                constants::G_SYMMETRY  * (ci == cj ? 1.f : -1.f);
                            ccpp *= temp;
                            ccpp += constants::H_COULOMB * Buffer::MeanField::rhc[mi];
                            ccpp *= mi_r;
                            grbb  = Buffer::MeanField::rbij[mi];

                            temp     = smem->ei[phid] + ej;
                            ccrr     = grbb * ccpp / temp;
                            betaij.x = (betaij.x + smem->px[phid]) / temp;
                            betaij.y = (betaij.y + smem->py[phid]) / temp;
                            betaij.z = (betaij.z + smem->pz[phid]) / temp;

                            // axis x
                            temp   = smem->rx[phid] - Buffer::Participant::position_x[pid];  // distance x, rij.x
                            cij    = betaij.x - smem->px[phid] / smem->ei[phid];
                            ffr.x += (temp + cij * grbb) * 2.f * ccrr;
                            ffp.x -= (temp + betaij.x * grbb) * 2.f * ccpp;

                            // axis y
                            temp   = smem->ry[phid] - Buffer::Participant::position_y[pid];
                            cij    = betaij.y - smem->py[phid] / smem->ei[phid];
                            ffr.y += (temp + cij * grbb) * 2.f * ccrr;
                            ffp.y -= (temp + betaij.y * grbb) * 2.f * ccpp;

                            // axis z
                            temp   = smem->rz[phid] - Buffer::Participant::position_z[pid];
                            cij    = betaij.z - smem->pz[phid] / smem->ei[phid];
                            ffr.z += (temp + cij * grbb) * 2.f * ccrr;
                            ffp.z -= (temp + betaij.z * grbb) * 2.f * ccpp;

                            assertNAN(ffr.x);
                            assertNAN(ffr.y);
                            assertNAN(ffr.z);
                            assertNAN(ffp.x);
                            assertNAN(ffp.y);
                            assertNAN(ffp.z);
                        }
                        __syncthreads();
                    }
                    
                    ffr.x = rsum32_(ffr.x);
                    ffr.y = rsum32_(ffr.y);
                    ffr.z = rsum32_(ffr.z);
                    ffp.x = rsum32_(ffp.x);
                    ffp.y = rsum32_(ffp.y);
                    ffp.z = rsum32_(ffp.z);
                    smem->ffrx[phid] = ffr.x;
                    smem->ffry[phid] = ffr.y;
                    smem->ffrz[phid] = ffr.z;
                    smem->ffpx[phid] = ffp.x;
                    smem->ffpy[phid] = ffp.y;
                    smem->ffpz[phid] = ffp.z;
                    __syncthreads();
                }

                pid = ib * CUDA_WARP_SIZE + warp_idx;
                if (pid < Buffer::model_cached->current_field_size) {
                    pid += Buffer::model_cached->offset_1d;
                    if (post_step) {
                        Buffer::MeanField::f0rx[pid] = smem->ffrx[warp_idx];
                        Buffer::MeanField::f0ry[pid] = smem->ffry[warp_idx];
                        Buffer::MeanField::f0rz[pid] = smem->ffrz[warp_idx];
                        Buffer::MeanField::f0px[pid] = smem->ffpx[warp_idx];
                        Buffer::MeanField::f0py[pid] = smem->ffpy[warp_idx];
                        Buffer::MeanField::f0pz[pid] = smem->ffpz[warp_idx];
                    }
                    else {
                        Buffer::MeanField::ffrx[pid] = smem->ffrx[warp_idx];
                        Buffer::MeanField::ffry[pid] = smem->ffry[warp_idx];
                        Buffer::MeanField::ffrz[pid] = smem->ffrz[warp_idx];
                        Buffer::MeanField::ffpx[pid] = smem->ffpx[warp_idx];
                        Buffer::MeanField::ffpy[pid] = smem->ffpy[warp_idx];
                        Buffer::MeanField::ffpz[pid] = smem->ffpz[warp_idx];
                    }
                }
                __syncthreads();
            }

            return;
        }


        __device__ float getPotential(int i, bool ci) {
            int lane_idx = threadIdx.x % CUDA_WARP_SIZE;
            int mi;  // matrix index (1d)

            float rha;
            float rhoa = 0.f;
            float rhos = 0.f;
            float rhoc = 0.f;
            float rho3;
            int   pid;
            bool  cj;
            for (int jw = 0; jw < Buffer::model_cached->iter_gwarp; ++jw) {
                int j = lane_idx + jw * CUDA_WARP_SIZE;
                if (i < Buffer::model_cached->current_field_size &&
                    j < Buffer::model_cached->current_field_size) {
                    mi    = i * Buffer::model_cached->current_field_size + j
                            + Buffer::model_cached->offset_2d;
                    pid   = (int)Buffer::model_cached->participant_idx[j] + Buffer::model_cached->offset_1d;
                    cj    = Buffer::Participant::flags[pid] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    rha   = Buffer::MeanField::rha[mi];
                    rhoa += rha;
                    rhoc += Buffer::MeanField::rhe[mi];
                    rha   = ci == cj ? rha : -rha;
                    rhos += rha;
                }
                __syncthreads();
            }
            rhoa = rsum32_(rhoa);
            rhoc = rsum32_(rhoc);
            rhos = rsum32_(rhos);
            rho3 = powf(rhoa, constants::H_SKYRME_GAMM);

            return (
                constants::H_SKYRME_C0 * rhoa +
                constants::H_SKYRME_C3 * rho3 +
                constants::H_SYMMETRY  * rhos +
                constants::H_COULOMB   * rhoc
            );
                   
        }


        __device__ void calTotalKineticEnergy() {
            bool leading_warp = !(threadIdx.x % CUDA_WARP_SIZE);
            Buffer::model_cached->ekinal = 0.f;

            __syncthreads();
            float temp;
            for (int i = 0; i <= Buffer::model_cached->current_field_size / blockDim.x; ++i) {
                int ip = i * blockDim.x + threadIdx.x;
                float mass   = 0.f;
                float energy = 0.f;
                if (ip < Buffer::model_cached->current_field_size) {
                    ip = (int)Buffer::model_cached->participant_idx[ip] + Buffer::model_cached->offset_1d;
                    mass    = Buffer::Participant::mass[ip];
                    //mass = Buffer::Participant::flags[ip] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON
                    //    ? constants::MASS_PROTON
                    //    : constants::MASS_NEUTRON;
                    energy  = mass * mass;
                    temp    = Buffer::Participant::momentum_x[ip];
                    energy += temp * temp;
                    temp    = Buffer::Participant::momentum_y[ip];
                    energy += temp * temp;
                    temp    = Buffer::Participant::momentum_z[ip];
                    energy += temp * temp;
                    energy  = sqrtf(energy) - mass;  // kinetic energy
                }
                __syncthreads();
                energy = rsum32_(energy);
                if (leading_warp)
                    atomicAdd(&Buffer::model_cached->ekinal, energy);
                __syncthreads();
            }
        }


        __device__ void calExcitationEnergyAndMomentum() {
            calTotalPotentialEnergy();

            int warp_idx = threadIdx.x % CUDA_WARP_SIZE;

            Buffer::model_cached->excitation     = 0.f;

            // net momentum in LAB
            float3 pcm0;
            float  mcm = 0.f;
            pcm0.x = 0.f;
            pcm0.y = 0.f;
            pcm0.z = 0.f;

            // using gradient buffer (ffp, rh3d) for saving memory
            for (int i = 0; i < Buffer::model_cached->iter_gwarp; ++i) {
                int ip = i * CUDA_WARP_SIZE + warp_idx;
                if (ip < Buffer::model_cached->current_field_size) {
                    ip = (int)Buffer::model_cached->participant_idx[ip] + Buffer::model_cached->offset_1d;
                    mcm    += Buffer::Participant::mass[ip];
                    pcm0.x += Buffer::Participant::momentum_x[ip];
                    pcm0.y += Buffer::Participant::momentum_y[ip];
                    pcm0.z += Buffer::Participant::momentum_z[ip];
                }
                __syncthreads();
            }
            mcm     = rsum32_(mcm);
            pcm0.x  = rsum32_(pcm0.x);
            pcm0.y  = rsum32_(pcm0.y);
            pcm0.z  = rsum32_(pcm0.z);
            

            // save to shared memory
            Buffer::model_cached->momentum_system[0] = pcm0.x;
            Buffer::model_cached->momentum_system[1] = pcm0.y;
            Buffer::model_cached->momentum_system[2] = pcm0.z;

            // boost vector
            float gamma = 
                pcm0.x * pcm0.x +
                pcm0.y * pcm0.y +
                pcm0.z * pcm0.z;
            float et = mcm * mcm + gamma;
            gamma    = 1.f / sqrtf(1.f - gamma / et);  // boost gamma
            et       = sqrtf(et);
            pcm0.x  /= et;  // boost beta
            pcm0.y  /= et;
            pcm0.z  /= et;
            et      /= (float)Buffer::model_cached->current_field_size;
            
            // CM system energy
            float  es = 0.f;
            float3 pcm;
            float  temp;
            for (int i = 0; i < Buffer::model_cached->iter_gwarp; ++i) {
                int ip = i * CUDA_WARP_SIZE + warp_idx;
                if (ip < Buffer::model_cached->current_field_size) {
                    ip     = (int)Buffer::model_cached->participant_idx[ip] + Buffer::model_cached->offset_1d;
                    pcm.x  = Buffer::Participant::momentum_x[ip];
                    pcm.y  = Buffer::Participant::momentum_y[ip];
                    pcm.z  = Buffer::Participant::momentum_z[ip];
                    temp   = pcm.x * pcm0.x + pcm.y * pcm0.y + pcm.z * pcm0.z;  // trans vector
                    temp   = gamma / (gamma + 1) * temp + et / gamma;
                    pcm.x -= temp * pcm0.x;
                    pcm.y -= temp * pcm0.y;
                    pcm.z -= temp * pcm0.z;
                    temp   = Buffer::Participant::mass[ip];
                    temp  *= temp;
                    temp  += pcm.x * pcm.x + pcm.y * pcm.y + pcm.z * pcm.z;
                    es    += sqrtf(temp);
                }
                __syncthreads();
            }
            es = rsum32_(es);

            // total invariant mass = nucleon mass + internal kinetic + potential
            int z = Buffer::model_cached->za_system.x;
            int a = Buffer::model_cached->za_system.y;
            float mass_inv = mass_table[z].get(a) * 1e-3f;
            float mass_nuc = ::constants::MASS_PROTON_GEV * z + ::constants::MASS_NEUTRON_GEV * (a - z);
            es += Buffer::model_cached->vtot;  // binding energy
            // excitation energy = invariant mass - observed mass
            
            Buffer::model_cached->mass_system = mass_inv;
            Buffer::model_cached->excitation  = fmax(0.f, es - mcm + mass_nuc - mass_inv);
        }


        __device__ void calTotalPotentialEnergy() {
            bool leading_warp = !(threadIdx.x %  CUDA_WARP_SIZE);
            int  lane_idx     = threadIdx.x >> CUDA_WARP_SHIFT;
            int  lane_dim     = blockDim.x  >> CUDA_WARP_SHIFT;

            Buffer::model_cached->vtot = 0.f;
            __syncthreads();

            for (int ib = 0; ib < Buffer::model_cached->iter_gblock; ++ib) {
                for (int iw = 0; iw < Buffer::model_cached->iter_glane; ++iw) {
                    int   i   = ib * CUDA_WARP_SIZE + iw * lane_dim + lane_idx;
                    bool  val = i < Buffer::model_cached->current_field_size;
                    int   pi  = val ? (int)Buffer::model_cached->participant_idx[i] : 0;
                    pi += Buffer::model_cached->offset_1d;
                    i   = val ? i : 0;
                    bool  ci  = Buffer::Participant::flags[pi] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    float vi  = getPotential(i, ci);
                    if (leading_warp && val)
                        atomicAdd(&Buffer::model_cached->vtot, vi);
                    __syncthreads();
                }
                __syncthreads();
            }
        }


        __device__ void prepareNuclei(bool target) {
            int z = (int)Buffer::model_cached->za_nuc[target].x;
            int a = (int)Buffer::model_cached->za_nuc[target].y;
            int offset = target ? (int)Buffer::model_cached->za_nuc[0].y : 0;
            assert(a + offset <= Buffer::MAX_DIMENSION_PARTICIPANT);
            __syncthreads();
            setDimension(a);
            NucleiSharedMem* smem = (NucleiSharedMem*)mcutil::cache_univ;
            smem->offset = offset;

            // Wood-Saxon parameters
            float radm = constants::WOOD_SAXON_RADIUS_00
                * powf((float)a, ::constants::ONE_OVER_THREE);
            float rt00 = radm - constants::WOOD_SAXON_RADIUS_01;
            radm -= constants::QMD_NUCLEUS_CUTOFF_A * (constants::H_SKYRME_GAMM - 1.0f)
                - constants::QMD_NUCLEUS_CUTOFF_B;
            float rmax = 1.f / (1.f + expf(-rt00 / constants::WOOD_SAXON_DIFFUSE));

            smem->radm = radm;
            smem->rt00 = rt00;
            smem->rmax = rmax;

            // energy
            float ebini = Buffer::model_cached->mass[target]
                - ::constants::MASS_PROTON_GEV * (float)z
                - ::constants::MASS_NEUTRON_GEV * (float)(a - z);
            ebini /= (float)a;  // binding energy per nucleon [GeV]
            float ebin0 = a == 4 ? ebini - 4.5e-3f : ebini - 1.5e-3f;
            float ebin1 = a == 4 ? ebini + 4.5e-3f : ebini + 1.5e-3f;

            smem->ebini = ebini;
            smem->ebin0 = ebin0;
            smem->ebin1 = ebin1;

            // set participant target tape & type
            for (int i = 0; i <= a / blockDim.x; ++i) {
                int   pp = i * blockDim.x + threadIdx.x;
                int   flag;
                float mass;
                if (pp < a) {
                    flag = pp < (int)Buffer::model_cached->za_nuc[target].x
                        ? INITIAL_FLAG_PROTON
                        : INITIAL_FLAG_NEUTRON;
                    mass = flag == INITIAL_FLAG_PROTON ? ::constants::MASS_PROTON_GEV : ::constants::MASS_NEUTRON_GEV;
                    flag |= target ? PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET : PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE;
                    Buffer::model_cached->participant_idx[pp] = (unsigned char)(pp + smem->offset);
                    pp += smem->offset + Buffer::model_cached->offset_1d;
                    Buffer::Participant::flags[pp] = flag;
                    Buffer::Participant::mass[pp]  = mass;
                }
                __syncthreads();
            }

        }


        __device__ bool sampleNuclei(bool target) {
            int a = (int)Buffer::model_cached->za_nuc[target].y;
            NucleiSharedMem* smem = (NucleiSharedMem*)mcutil::cache_univ;

            // initialize momentum
            for (int i = 0; i <= a / blockDim.x; ++i) {
                int pp = i * blockDim.x + threadIdx.x;
                if (pp < a) {
                    pp += smem->offset + Buffer::model_cached->offset_1d;
                    Buffer::Participant::momentum_x[pp] = 0.f;
                    Buffer::Participant::momentum_y[pp] = 0.f;
                    Buffer::Participant::momentum_z[pp] = 0.f;
                }
                __syncthreads();
            }

            // sample loop
            if (!sampleNucleonPosition(target, smem))
                return false;
            __syncthreads();

            cal2BodyQuantities();

            if (!sampleNucleonMomentum(target, smem))
                return false;
            __syncthreads();

            // post-processing

            killCMMotion(target, smem->offset);
            killAngularMomentum(target, smem->offset);
            __syncthreads();

            // check binding energy
            cal2BodyQuantities();
            calTotalKineticEnergy();
            calTotalPotentialEnergy();

            smem->ebinal = (Buffer::model_cached->vtot + Buffer::model_cached->ekinal) 
                / (float)Buffer::model_cached->za_nuc[target].y;

            __syncthreads();
            if (smem->ebinal < smem->ebin0 || smem->ebinal > smem->ebin1)  // internal energy violation
                return false;

            // using INCL off-shell algorithm for energy-momentum conservation
            // forcingConservationLaw(target, smem->offset);

            // initialize nuclei position & momentum
            setNucleiPhaseSpace(target, smem->offset);

            return true;
        }


        __device__ void sampleNucleon(bool target) {
            int z = (int)Buffer::model_cached->za_nuc[target].x;
            int a = (int)Buffer::model_cached->za_nuc[target].y;
            int offset = target ? (int)Buffer::model_cached->za_nuc[0].y : 0;
            assert(a + offset <= Buffer::MAX_DIMENSION_PARTICIPANT);
            __syncthreads();
            setDimension(a);

            // set participant target tape & type
            int   flag = z == a
                ? INITIAL_FLAG_PROTON
                : INITIAL_FLAG_NEUTRON;
            float mass = flag == INITIAL_FLAG_PROTON 
                ? ::constants::MASS_PROTON_GEV 
                : ::constants::MASS_NEUTRON_GEV;
            flag |= target 
                ? PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET 
                : PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE;

            int pp = offset + Buffer::model_cached->offset_1d;
            Buffer::Participant::flags[pp]      = flag;
            Buffer::Participant::mass[pp]       = mass;
            Buffer::Participant::position_x[pp] = 0.f;
            Buffer::Participant::position_y[pp] = 0.f;
            Buffer::Participant::position_z[pp] = 0.f;
            Buffer::Participant::momentum_x[pp] = 0.f;
            Buffer::Participant::momentum_y[pp] = 0.f;
            Buffer::Participant::momentum_z[pp] = 0.f;
            __syncthreads();

            // initialize nucleon position & momentum
            setNucleiPhaseSpace(target, offset);
        }


        __device__ bool sampleNucleonPosition(bool target, NucleiSharedMem* smem) {
            int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int warp_idx   = threadIdx.x %  CUDA_WARP_SIZE;
            bool exit_condition;

            Buffer::model_cached->nuc_counter.z = 0u;  // local counter, 2nd
            smem->ia = 0;
            __syncthreads();

            while (true) {
                if (!threadIdx.x)
                    Buffer::model_cached->nuc_counter.z++;
                smem->ib = 0;

                if (!threadIdx.x)
                    smem->condition_broadcast = smem->ia >= Buffer::model_cached->current_field_size;
                __syncthreads();
                exit_condition = smem->condition_broadcast;
                __syncthreads();
                if (exit_condition)
                    break;

                if (!threadIdx.x)
                    smem->condition_broadcast = Buffer::model_cached->nuc_counter.z == UCHAR_MAX;
                __syncthreads();
                exit_condition = smem->condition_broadcast;
                __syncthreads();
                if (exit_condition)
                    return false;

                {   // nucleon sampler
                    float sint, cost, sinp, cosp;
                    int   it = 99999;
                    float rrr = powf(curand_uniform(&rand_state[thread_idx]), ::constants::ONE_OVER_THREE);
                    rrr *= smem->radm;
                    float rwod = 1.f / (1.f + expf((rrr - smem->rt00) / constants::WOOD_SAXON_DIFFUSE));
                    cost = 2.f * curand_uniform(&rand_state[thread_idx]) - 1.f;
                    sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
                    sincosf(::constants::FP32_TWO_PI * curand_uniform(&rand_state[thread_idx]), &sinp, &cosp);

                    if (curand_uniform(&rand_state[thread_idx]) * smem->rmax > rwod)
                        it = atomicAdd(&smem->ib, 1);

                    if (it < CUDA_WARP_SIZE) {  // maximum candidate = 32
                        smem->y[it] = sint * sinp * rrr;
                        smem->x[it] = sint * cosp * rrr;
                        smem->z[it] = cost * rrr;
                    }
                }
                __syncthreads();

                // check distance to others
                int iter_max = min(smem->ib, CUDA_WARP_SIZE);
                __syncthreads(); 
                for (int m = 0; m < iter_max; ++m) {
                    if (!threadIdx.x)
                        smem->condition_broadcast = smem->ia >= Buffer::model_cached->current_field_size;
                    __syncthreads();
                    exit_condition = smem->condition_broadcast;
                    __syncthreads();
                    if (exit_condition)
                        break;

                    // set nucleon type
                    int flag = smem->ia < Buffer::model_cached->za_nuc[target].x
                        ? INITIAL_FLAG_PROTON
                        : INITIAL_FLAG_NEUTRON;
                    smem->blocked = false;

                    __syncthreads();
                    for (int i = 0; i <= smem->ia / blockDim.x; ++i) {
                        int ip = i * blockDim.x + threadIdx.x;
                        if (ip < smem->ia) {
                            ip += Buffer::model_cached->offset_1d + smem->offset;
                            float dmin2 = (Buffer::Participant::flags[ip] == flag)
                                ? constants::NUCLEUS_MD2_ISO_2
                                : constants::NUCLEUS_MD2_ISO_0;
                            float dd2 = 0.f;
                            float dd;

                            // x-axis
                            dd = Buffer::Participant::position_x[ip] - smem->x[m];
                            dd2 += dd * dd;

                            // y-axis
                            dd = Buffer::Participant::position_y[ip] - smem->y[m];
                            dd2 += dd * dd;

                            // z-axis
                            dd = Buffer::Participant::position_z[ip] - smem->z[m];
                            dd2 += dd * dd;

                            if (dd2 < dmin2)
                                smem->blocked = true;
                        }
                        __syncthreads();

                        if (!threadIdx.x)
                            smem->condition_broadcast = smem->blocked;
                        __syncthreads();
                        exit_condition = smem->condition_broadcast;
                        __syncthreads();
                        if (exit_condition)
                            break;
                    }

                    if (!threadIdx.x) {
                        if (!smem->blocked) {
                            int ip = Buffer::model_cached->offset_1d + smem->offset + smem->ia;
                            Buffer::Participant::position_x[ip] = smem->x[m];
                            Buffer::Participant::position_y[ip] = smem->y[m];
                            Buffer::Participant::position_z[ip] = smem->z[m];
                            smem->ia++;
                        }
                    }
                    __syncthreads();
                }
            }

            __syncthreads();
            return true;
        }


        __device__ bool sampleNucleonMomentum(bool target, NucleiSharedMem* smem) {
            int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int warp_idx   = threadIdx.x % CUDA_WARP_SIZE;
            bool exit_condition;

            bool prepare_sampling_coeff = true;
            Buffer::model_cached->nuc_counter.z = 0u;  // local counter, 2nd
            smem->ia = 0;
            __syncthreads();

            // initialize phase-space cumulative data
            for (int i = 0; i <= Buffer::model_cached->current_field_size / blockDim.x; ++i) {
                int ip = i * blockDim.x + threadIdx.x;
                if (ip < Buffer::model_cached->current_field_size)
                    smem->phg[ip] = 0.f;
                __syncthreads();
            }

            // while (smem->ia < Buffer::model_cached->current_field_size) {
            while (true) {
                if (!threadIdx.x)
                    Buffer::model_cached->nuc_counter.z++;
                smem->ib = 0;

                if (!threadIdx.x)
                    smem->condition_broadcast = smem->ia >= Buffer::model_cached->current_field_size;
                __syncthreads();
                exit_condition = smem->condition_broadcast;
                __syncthreads();
                if (exit_condition)
                    break;

                if (!threadIdx.x)
                    smem->condition_broadcast = Buffer::model_cached->nuc_counter.z == UCHAR_MAX;
                __syncthreads();
                exit_condition = smem->condition_broadcast;
                __syncthreads();
                if (exit_condition)
                    return false;

                if (prepare_sampling_coeff) {
                    int ci = smem->ia + Buffer::model_cached->offset_1d + smem->offset;
                    int cj;
                    ci     = Buffer::Participant::flags[ci] 
                        & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    float rho_a = 0.f;
                    float rho_s = 0.f;
                    float rho_c = 0.f;
                    for (int j = 0; j < Buffer::model_cached->iter_gwarp; ++j) {
                        int   ja = j * CUDA_WARP_SIZE + warp_idx;
                        float temp;
                        if (ja < Buffer::model_cached->current_field_size) {
                            int mi = smem->ia * Buffer::model_cached->current_field_size + ja + Buffer::model_cached->offset_2d;
                            cj = (int)Buffer::model_cached->participant_idx[ja];
                            cj = Buffer::Participant::flags[cj + Buffer::model_cached->offset_1d]
                                & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                            temp   = Buffer::MeanField::rha[mi];
                            rho_a += temp;
                            temp   = ci == cj ? temp : -temp;
                            rho_s += temp;
                            rho_c += Buffer::MeanField::rhc[mi];
                        }
                        __syncthreads();
                    }
                    rho_a = rsum32_(rho_a);
                    rho_s = rsum32_(rho_s);
                    rho_c = rsum32_(rho_c);
                    float rhol = constants::GS_CD * (rho_a + 1.f);
                    float pfm  = ::constants::FP32_HBARC
                        * powf(::constants::THREE_OVER_TWO * ::constants::FP32_PI_SQ * rhol, ::constants::ONE_OVER_THREE);
                    pfm = smem->ia > 10 && smem->ebini > -5.5e-3f
                        ? pfm * (1.f + 0.2f * sqrtf(fabsf(8e-3f + smem->ebini) / 8e-3f))
                        : pfm;
                    smem->pfm  = pfm;
                    smem->dpot =
                        constants::GS_C0 * rho_a +
                        constants::GS_C3 * powf(rho_a, constants::H_SKYRME_GAMM) +
                        constants::GS_CS * rho_s +
                        constants::GS_CL * rho_c;
                    prepare_sampling_coeff = false;
                }
                __syncthreads();

                // set nucleon type
                int   flag  = smem->ia < Buffer::model_cached->za_nuc[target].x
                    ? INITIAL_FLAG_PROTON
                    : INITIAL_FLAG_NEUTRON;
                float mass = flag == INITIAL_FLAG_PROTON
                    ? ::constants::MASS_PROTON_GEV
                    : ::constants::MASS_NEUTRON_GEV;

                // sample nucleon momentum (atomic shared)
                {   // nucleon sampler
                    int   it  = 99999;
                    float ppp = powf(curand_uniform(&rand_state[thread_idx]), ::constants::ONE_OVER_THREE);
                    ppp *= smem->pfm;
                    float ke  = sqrtf(ppp * ppp + mass * mass) - mass;
                    if (ke <= -smem->dpot)
                        it = atomicAdd(&smem->ib, 1);
                    __syncthreads();

                    if (it < CUDA_WARP_SIZE)
                        smem->z[it] = ppp;
                }
                __syncthreads();

                int iter_max = min(smem->ib, CUDA_WARP_SIZE);
                __syncthreads();

                // sample momentum direction
                if (threadIdx.x < iter_max) {
                    float sint, cost, sinp, cosp;
                    cost = 2.f * curand_uniform(&rand_state[thread_idx]) - 1.f;
                    sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
                    sincosf(::constants::FP32_TWO_PI * curand_uniform(&rand_state[thread_idx]), &sinp, &cosp);

                    smem->x[warp_idx] = sint * sinp * smem->z[warp_idx];
                    smem->y[warp_idx] = sint * cosp * smem->z[warp_idx];
                    smem->z[warp_idx] = cost * smem->z[warp_idx];
                }
                __syncthreads();
                // pauli principle
                for (int m = 0; m < iter_max; ++m) {
                    smem->blocked  = false;
                    smem->ps_cumul = 0.f;
                    __syncthreads();
                    for (int i = 0; i <= smem->ia / blockDim.x; ++i) {
                        int   ip = i * blockDim.x + threadIdx.x;
                        int   mi = m * Buffer::model_cached->current_field_size + ip + Buffer::model_cached->offset_2d;
                        float ps = 0.f;
                        float temp;
                        if (ip < smem->ia) {
                            ip += Buffer::model_cached->offset_1d + smem->offset;
                            float expa = -Buffer::MeanField::rr2[mi] * constants::PAULI_CPW;
                            // momentum space distance
                            temp = Buffer::Participant::momentum_x[ip] - smem->x[m];
                            ps = temp * temp;
                            temp = Buffer::Participant::momentum_y[ip] - smem->y[m];
                            ps += temp * temp;
                            temp = Buffer::Participant::momentum_z[ip] - smem->z[m];
                            ps += temp * temp;
                            expa = expa - ps * constants::PAULI_CPH;
                            ps = expa <= constants::PAULI_EPSX ? 0.f : expf(expa);
                            if (Buffer::Participant::flags[ip] != flag)
                                ps = 0.f;
                            ip = i * blockDim.x + threadIdx.x;
                            if (ps * constants::PAULI_CPC > 0.2f)
                                smem->blocked = true;
                            if ((ps + smem->phg[ip]) * constants::PAULI_CPC > 0.5f)
                                smem->blocked = true;
                        }
                        __syncthreads();

                        if (!threadIdx.x)
                            smem->condition_broadcast = smem->blocked;
                        __syncthreads();
                        exit_condition = smem->condition_broadcast;
                        __syncthreads();
                        if (exit_condition)
                            break;

                        ps = rsum32_(ps);
                        if (!warp_idx)
                            atomicAdd(&smem->ps_cumul, ps);
                        __syncthreads();

                        if (!threadIdx.x) {
                            if (smem->ps_cumul * constants::PAULI_CPC > 0.3f)
                                smem->blocked = true;
                            smem->condition_broadcast = smem->blocked;
                        } 
                        __syncthreads();
                        exit_condition = smem->condition_broadcast;
                        __syncthreads();
                        if (exit_condition)
                            break;
                    }

                    __syncthreads();

                    if (!threadIdx.x)
                        smem->condition_broadcast = smem->blocked;
                    __syncthreads();
                    exit_condition = smem->condition_broadcast;
                    __syncthreads();
                    if (exit_condition)
                        continue;

                    if (!threadIdx.x)
                        smem->phg[m] = smem->ps_cumul;
                    __syncthreads();
                    
                    exit_condition = smem->ia == Buffer::model_cached->current_field_size - 1;
                    __syncthreads();
                    if (!exit_condition) {
                        for (int i = 0; i <= smem->ia / blockDim.x; ++i) {
                            int   ip = i * blockDim.x + threadIdx.x;
                            int   mi = m * Buffer::model_cached->current_field_size + ip + Buffer::model_cached->offset_2d;
                            float ps = 0.f;
                            float temp;
                            if (ip < smem->ia) {
                                ip += Buffer::model_cached->offset_1d + smem->offset;
                                float expa = -Buffer::MeanField::rr2[mi] * constants::PAULI_CPW;
                                // momentum space distance
                                temp = Buffer::Participant::momentum_x[ip] - smem->x[m];
                                ps  += temp * temp;
                                temp = Buffer::Participant::momentum_y[ip] - smem->y[m];
                                ps  += temp * temp;
                                temp = Buffer::Participant::momentum_z[ip] - smem->z[m];
                                ps  += temp * temp;
                                expa = expa - ps * constants::PAULI_CPH;
                                ps   = expf(expa);
                                if (Buffer::Participant::flags[ip] != flag)
                                    ps = 0.f;
                                ip = i * blockDim.x + threadIdx.x;
                                smem->phg[ip] += ps;
                            }
                            __syncthreads();
                        }
                    }
                    if (!threadIdx.x) {
                        int ip = Buffer::model_cached->offset_1d + smem->offset + smem->ia;

                        assertNAN(smem->x[m]);
                        assertNAN(smem->y[m]);
                        assertNAN(smem->z[m]);

                        Buffer::Participant::momentum_x[ip] = smem->x[m];
                        Buffer::Participant::momentum_y[ip] = smem->y[m];
                        Buffer::Participant::momentum_z[ip] = smem->z[m];
                        smem->ia++;
                    }
                    __syncthreads();
                    prepare_sampling_coeff = true;
                    break;
                }
            }

            return true;
        }


        __device__ void killCMMotion(bool target, int offset) {
            int warp_idx = threadIdx.x % CUDA_WARP_SIZE;
            float* rpcm_shared = &mcutil::cache_univ[NUCLEI_SHARED_MEM_OFFSET];
            {
                float3 pcm;
                float3 rcm;
                float  mcm = 0.f;
                pcm.x = 0.f;
                pcm.y = 0.f;
                pcm.z = 0.f;
                rcm.x = 0.f;
                rcm.y = 0.f;
                rcm.z = 0.f;

                for (int i = 0; i < Buffer::model_cached->iter_gwarp; ++i) {
                    int ip = i * CUDA_WARP_SIZE + warp_idx;
                    if (ip < Buffer::model_cached->current_field_size) {
                        ip += Buffer::model_cached->offset_1d + offset;
                        float mass = Buffer::Participant::flags[ip] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON
                            ? ::constants::MASS_PROTON_GEV
                            : ::constants::MASS_NEUTRON_GEV;
                        pcm.x += Buffer::Participant::momentum_x[ip];
                        pcm.y += Buffer::Participant::momentum_y[ip];
                        pcm.z += Buffer::Participant::momentum_z[ip];
                        rcm.x += Buffer::Participant::position_x[ip] * mass;
                        rcm.y += Buffer::Participant::position_y[ip] * mass;
                        rcm.z += Buffer::Participant::position_z[ip] * mass;
                        mcm += mass;
                    }
                    __syncthreads();
                }
                pcm.x = rsum32_(pcm.x);
                pcm.y = rsum32_(pcm.y);
                pcm.z = rsum32_(pcm.z);
                rcm.x = rsum32_(rcm.x);
                rcm.y = rsum32_(rcm.y);
                rcm.z = rsum32_(rcm.z);
                mcm   = rsum32_(mcm);

                float a = (int)Buffer::model_cached->za_nuc[target].y;
                pcm.x /= a;
                pcm.y /= a;
                pcm.z /= a;
                rcm.x /= mcm;
                rcm.y /= mcm;
                rcm.z /= mcm;

                rpcm_shared[0] = pcm.x;
                rpcm_shared[1] = pcm.y;
                rpcm_shared[2] = pcm.z;
                rpcm_shared[3] = rcm.x;
                rpcm_shared[4] = rcm.y;
                rpcm_shared[5] = rcm.z;
            }
            __syncthreads();

            // move to CM system & calculate angular momentum of system
            float3 ll;
            ll.x = 0.f;
            ll.y = 0.f;
            ll.z = 0.f;
            for (int i = 0; i <= Buffer::model_cached->current_field_size / blockDim.x; ++i) {
                int ip = i * blockDim.x + threadIdx.x;
                float3 p;
                float3 r;
                if (ip < Buffer::model_cached->current_field_size) {
                    ip += Buffer::model_cached->offset_1d + offset;
                    p.x = Buffer::Participant::momentum_x[ip];
                    p.y = Buffer::Participant::momentum_y[ip];
                    p.z = Buffer::Participant::momentum_z[ip];
                    r.x = Buffer::Participant::position_x[ip];
                    r.y = Buffer::Participant::position_y[ip];
                    r.z = Buffer::Participant::position_z[ip];
                    p.x -= rpcm_shared[0];
                    p.y -= rpcm_shared[1];
                    p.z -= rpcm_shared[2];
                    r.x -= rpcm_shared[3];
                    r.y -= rpcm_shared[4];
                    r.z -= rpcm_shared[5];
                    Buffer::Participant::momentum_x[ip] = p.x;
                    Buffer::Participant::momentum_y[ip] = p.y;
                    Buffer::Participant::momentum_z[ip] = p.z;
                    assertNAN(p.x);
                    assertNAN(p.y);
                    assertNAN(p.z);
                    Buffer::Participant::position_x[ip] = r.x;
                    Buffer::Participant::position_y[ip] = r.y;
                    Buffer::Participant::position_z[ip] = r.z;
                    ll.x += r.y * p.z - r.z * p.y;
                    ll.y += r.z * p.x - r.x * p.z;
                    ll.z += r.x * p.y - r.y * p.x;
                }
                __syncthreads();
            }

            ll.x = rsum32_(ll.x);
            ll.y = rsum32_(ll.y);
            ll.z = rsum32_(ll.z);
            rpcm_shared[0] = 0.f;
            rpcm_shared[1] = 0.f;
            rpcm_shared[2] = 0.f;
            __syncthreads();

            if (!warp_idx) {
                atomicAdd(&rpcm_shared[0], ll.x);
                atomicAdd(&rpcm_shared[1], ll.y);
                atomicAdd(&rpcm_shared[2], ll.z);
            }
            __syncthreads();

            return;
        }


        __device__ void killAngularMomentum(bool target, int offset) {
            bool warp_lead = !(threadIdx.x %  CUDA_WARP_SIZE);
            // int  lane_idx  = threadIdx.x >> CUDA_WARP_SHIFT;
            // int  lane_dim  = blockDim.x  >> CUDA_WARP_SHIFT;
            float* ll  = &mcutil::cache_univ[NUCLEI_SHARED_MEM_OFFSET];
            float* opl = ll + 3;
            opl[0] = 0.f;
            opl[1] = 0.f;
            opl[2] = 0.f;

            // 3x3 block
            float* rr  = ll + 6;   // rr shared
            float* ss  = ll + 15;

            if (threadIdx.x < 9) {
                rr[threadIdx.x] = 0.f;
                ss[threadIdx.x] = !(threadIdx.x % 4) ? 1.f : 0.f;
            }
            __syncthreads();
            
            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                float3 r;
                float  rr_seg;
                r.x = 0.f;
                r.y = 0.f;
                r.z = 0.f;
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d + offset;
                    r.x = Buffer::Participant::position_x[i];
                    r.y = Buffer::Participant::position_y[i];
                    r.z = Buffer::Participant::position_z[i];
                }
                __syncthreads();
                // [0, 0]
                rr_seg = r.y * r.y + r.z * r.z;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[0], rr_seg);
                __syncthreads();
                // [1, 1]
                rr_seg = r.z * r.z + r.x * r.x;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[4], rr_seg);
                __syncthreads();
                // [2, 2]
                rr_seg = r.x * r.x + r.y * r.y;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[8], rr_seg);
                __syncthreads();
                // [0, 1] == [1, 0]
                rr_seg = -r.x * r.y;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[1], rr_seg);
                __syncthreads();
                // [2, 0]
                rr_seg = -2.f * r.z * r.x;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[6], rr_seg);
                __syncthreads();
                // [2, 1]
                rr_seg = -2.f * r.y * r.z;
                rr_seg = rsum32_(rr_seg);
                if (warp_lead)
                    atomicAdd(&rr[7], rr_seg);
                __syncthreads();
            }
            // [0, 1] == [1, 0]
            rr[3] = rr[1];
            __syncthreads();

            assertNAN(rr[0]);
            assertNAN(rr[4]);
            assertNAN(rr[8]);
            assertNAN(rr[1]);
            assertNAN(rr[6]);
            assertNAN(rr[7]);

            // inv
            if (!threadIdx.x) {
                for (int i = 0; i < 3; ++i) {
                    float x = rr[4 * i];
                    for (int j = 0; j < 3; ++j) {
                        rr[i * 3 + j] /= x;
                        ss[i * 3 + j] /= x;
                    }
                    for (int j = 0; j < 3; ++j) {
                        if (j != i) {
                            float y = rr[j * 3 + i];
                            for (int k = 0; k < 3; ++k) {
                                rr[j * 3 + k] += -y * rr[i * 3 + k];
                                ss[j * 3 + k] += -y * ss[i * 3 + k];
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // check whether inverse matrix is exist
            float rrx = 0.f;
            float ssx = 0.f;
            if (threadIdx.x % CUDA_WARP_SIZE < 9) {
                rrx = rr[threadIdx.x % CUDA_WARP_SIZE];
                ssx = ss[threadIdx.x % CUDA_WARP_SIZE];
            }
                
            if (__any_sync(0xffffffff, isnan(rrx)))
                return;

            if (__any_sync(0xffffffff, isnan(ssx)))
                return;

            // opl
            if (threadIdx.x < 3) {
                float tp = 0.f;
                tp += ss[threadIdx.x * 3]     * ll[0];
                tp += ss[threadIdx.x * 3 + 1] * ll[1];
                tp += ss[threadIdx.x * 3 + 2] * ll[2];
                opl[threadIdx.x] = tp;
            }
            __syncthreads();

            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                float3 r;
                float  rc;
                float  rr_seg;
                r.x = 0.f;
                r.y = 0.f;
                r.z = 0.f;
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    i  += Buffer::model_cached->offset_1d + offset;
                    r.x = Buffer::Participant::position_x[i];
                    r.y = Buffer::Participant::position_y[i];
                    r.z = Buffer::Participant::position_z[i];
                    // x 
                    rc = r.y * opl[2] - r.z * opl[1];
                    Buffer::Participant::momentum_x[i] += rc;
                    assertNAN(rc);
                    // y
                    rc = r.z * opl[0] - r.x * opl[2];
                    Buffer::Participant::momentum_y[i] += rc;
                    assertNAN(rc);
                    // z 
                    rc = r.x * opl[1] - r.y * opl[0];
                    Buffer::Participant::momentum_z[i] += rc;
                    assertNAN(rc);
                }
                __syncthreads();
            }

            return;
        }


        __device__ void forcingConservationLaw(bool target, int offset) {
            float  m1 = Buffer::model_cached->mass[target] - (Buffer::model_cached->vtot + Buffer::model_cached->ekinal);
            short2 za = Buffer::model_cached->za_nuc[target];
            float  m2 = ::constants::MASS_PROTON_GEV * (float)za.x + ::constants::MASS_NEUTRON_GEV * (float)(za.y - za.x);
            float  mr = m1 / m2;
            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    i    += Buffer::model_cached->offset_1d + offset;
                    Buffer::Participant::mass[i] *= mr;
                }
                __syncthreads();
            }
        }


        __device__ void setNucleiPhaseSpace(bool target, int offset) {
            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                int   i = ib * blockDim.x + threadIdx.x;
                float temp;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d + offset;
                    
                    // position
                    Buffer::Participant::position_x[i] += Buffer::model_cached->cc_rx[target];
                    temp = Buffer::Participant::position_z[i];
                    Buffer::Participant::position_z[i] 
                        = temp / Buffer::model_cached->cc_gamma[target] + Buffer::model_cached->cc_rz[target];

                    // momentum
                    Buffer::Participant::momentum_x[i] += Buffer::model_cached->cc_px[target];
                    temp = Buffer::Participant::momentum_z[i];
                    Buffer::Participant::momentum_z[i] 
                        = temp * Buffer::model_cached->cc_gamma[target] + Buffer::model_cached->cc_pz[target];
                }
                __syncthreads();
            }
        }


        __device__ void doPropagate() {
            // propagation main
            calGraduate(true);

            // 1st step
            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                int   i = ib * blockDim.x + threadIdx.x;
                float fft;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d;
                    // position
                    fft = Buffer::MeanField::f0rx[i];
                    Buffer::Participant::position_x[i] += constants::PROPAGATE_DT3 * fft;
                    fft = Buffer::MeanField::f0ry[i];
                    Buffer::Participant::position_y[i] += constants::PROPAGATE_DT3 * fft;
                    fft = Buffer::MeanField::f0rz[i];
                    Buffer::Participant::position_z[i] += constants::PROPAGATE_DT3 * fft;
                    // momentum
                    fft = Buffer::MeanField::f0px[i];
                    Buffer::Participant::momentum_x[i] += constants::PROPAGATE_DT3 * fft;
                    fft = Buffer::MeanField::f0py[i];
                    Buffer::Participant::momentum_y[i] += constants::PROPAGATE_DT3 * fft;
                    fft = Buffer::MeanField::f0pz[i];
                    Buffer::Participant::momentum_z[i] += constants::PROPAGATE_DT3 * fft;
                }
                __syncthreads();
            }

            cal2BodyQuantities();
            calGraduate();

            // 2nd step
            for (int ib = 0; ib <= Buffer::model_cached->current_field_size / blockDim.x; ++ib) {
                int   i = ib * blockDim.x + threadIdx.x;
                float fft;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d;
                    // position
                    fft  = Buffer::MeanField::f0rx[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffrx[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::position_x[i] += fft;
                    fft  = Buffer::MeanField::f0ry[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffry[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::position_y[i] += fft;
                    fft  = Buffer::MeanField::f0rz[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffrz[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::position_z[i] += fft;
                    // momentum
                    fft  = Buffer::MeanField::f0px[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffpx[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::momentum_x[i] += fft;
                    fft  = Buffer::MeanField::f0py[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffpy[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::momentum_y[i] += fft;
                    fft  = Buffer::MeanField::f0pz[i] * constants::PROPAGATE_DT1;
                    fft += Buffer::MeanField::ffpz[i] * constants::PROPAGATE_DT2;
                    Buffer::Participant::momentum_z[i] += fft;
                }
                __syncthreads();
            }

            cal2BodyQuantities(true);
        }


        __device__ void doClusterJudgement() {
            int warp_idx = threadIdx.x %  CUDA_WARP_SIZE;
            int lane_idx = threadIdx.x >> CUDA_WARP_SHIFT;
            int lane_dim = blockDim.x  >> CUDA_WARP_SHIFT;
            int mai;

            cal2BodyQuantities(false);

            ClusterSharedMem* smem = (ClusterSharedMem*)mcutil::cache_univ;

            // initialize cluster idx & number of cluster
            smem->n_cluster = 0;
            for (int ib = 0; ib < Buffer::model_cached->iter_1body; ++ib) {
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size)
                    smem->cluster_idx[i] = -1;
                __syncthreads();
            }

            // rhoa initialization
            float rhoa;
            for (int ib = 0; ib < Buffer::model_cached->iter_grh3d; ++ib) {
                int i = lane_idx + ib * lane_dim;
                rhoa = 0.f;
                for (int jw = 0; jw < Buffer::model_cached->iter_gwarp; ++jw) {
                    int j = warp_idx + jw * CUDA_WARP_SIZE;
                    if (i < Buffer::model_cached->current_field_size &&
                        j < Buffer::model_cached->current_field_size) {
                        mai = i * Buffer::model_cached->current_field_size + j
                            + Buffer::model_cached->offset_2d;
                        rhoa += Buffer::MeanField::rha[mai];
                    }
                    __syncthreads();
                }
                rhoa = rsum32_(rhoa);
                rhoa = powf(rhoa + 1, constants::G_SKYRME_GAMM);
                if (i < Buffer::model_cached->current_field_size)
                    smem->rhoa[i] = rhoa;
                __syncthreads();
            }
            
            // clustering
            for (int i = 0; i < Buffer::model_cached->current_field_size - 1; ++i) {
                int current_cluster   = smem->cluster_idx[i];
                smem->neighbor_found  = 0;
                smem->cluster_found   = -1;
                __syncthreads();

                // neighbor check
                for (int jb = 0; jb < Buffer::model_cached->iter_1body; ++jb) {
                    int j = jb * blockDim.x + threadIdx.x;
                    mai = i * Buffer::model_cached->current_field_size + j
                        + Buffer::model_cached->offset_2d;
                    bool neighb = false;
                    if (j < Buffer::model_cached->current_field_size && j > i) {
                        float rdist2 = Buffer::MeanField::rr2[mai];
                        float pdist2 = Buffer::MeanField::pp2[mai];
                        float pcc2   = smem->rhoa[i] + smem->rhoa[j];
                        pcc2 = pcc2 * pcc2 * constants::CLUSTER_CPF2;
                        // Check phase space: close enough?
                        neighb = rdist2 < constants::CLUSTER_R2 && pdist2 < pcc2;
                        if (current_cluster >= 0 && smem->cluster_idx[j] == current_cluster)
                            neighb = false;
                        if (neighb) {
                            smem->neighbor_found = true;
                            int cluster = (int)smem->cluster_idx[j];
                            if (cluster >= 0)
                                atomicCAS(&smem->cluster_found, -1, cluster);  // unique cluster
                        }
                    }
                    smem->neighbor[j] = neighb;
                    __syncthreads();
                }

                // sync
                int  neighbor_found = smem->neighbor_found;
                int  cluster_found  = smem->cluster_found;   // parent cluster of primary candidate 'j'
                __syncthreads();
                if (!neighbor_found)
                    continue;

                if (cluster_found >= 0) {
                    if (current_cluster < 0) { // no cluster i
                        if (!threadIdx.x)
                            smem->cluster_idx[i] = cluster_found;
                        __syncthreads();
                    }
                    else {
                        if (current_cluster < cluster_found) {
                            // using neighbor_found as temp
                            neighbor_found  = cluster_found;
                            cluster_found   = current_cluster;
                            current_cluster = neighbor_found;
                        }
                        __syncthreads();
                        for (int jb = 0; jb < Buffer::model_cached->iter_1body; ++jb) {
                            int j = jb * blockDim.x + threadIdx.x;
                            if (j < Buffer::model_cached->current_field_size) {
                                if (smem->cluster_idx[j] == current_cluster)
                                    smem->cluster_idx[j] = cluster_found;
                            }
                            __syncthreads();
                        }
                    }
                    i -= 1;
                    continue;  // retry i
                }
                else {  // no cluster j
                    if (current_cluster < 0) { // no cluster i
                        current_cluster = smem->n_cluster;
                        __syncthreads();
                        if (!threadIdx.x) {
                            smem->cluster_idx[i] = current_cluster;
                            smem->n_cluster      = current_cluster + 1;
                        }
                    }
                    __syncthreads();
                    for (int jb = 0; jb < Buffer::model_cached->iter_1body; ++jb) {
                        int j = jb * blockDim.x + threadIdx.x;
                        if (j < Buffer::model_cached->current_field_size) {
                            if (smem->neighbor[j])
                                smem->cluster_idx[j] = current_cluster;
                        }
                        __syncthreads();
                    }
                }
                __syncthreads();
            }

            // write to global memory
            Buffer::model_cached->n_cluster = smem->n_cluster;
            for (int ib = 0; ib < Buffer::model_cached->iter_1body; ++ib) {
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    int cid   = (int)smem->cluster_idx[i];
                    i += Buffer::model_cached->offset_1d;
                    int flags = Buffer::Participant::flags[i];
                    flags &= PARTICIPANT_FLAGS_MASK;
                    flags &= ~PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER;
                    if (cid >= 0) {
                        flags |= PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER;
                        flags |= cid << CLUSTER_IDX_SHIFT;
                    }
                    Buffer::Participant::flags[i] = flags;
                }
                __syncthreads();
            }
        }


        __device__ void checkFieldIntegrity() {
            for (int ib = 0; ib < Buffer::model_cached->iter_1body; ++ib) {
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d;
                    float px    = Buffer::Participant::momentum_x[i];
                    float py    = Buffer::Participant::momentum_y[i];
                    float pz    = Buffer::Participant::momentum_z[i];
                    float mass  = Buffer::Participant::mass[i];
                    if (isnan(px) || isnan(py) ||isnan(pz) || isnan(mass))
                        Buffer::model_cached->initial_flags.qmd.fmask |= MODEL_FLAGS::MODEL_FAIL_PROPAGATION;
                }
                __syncthreads();
            }
        }


        __device__ bool calculateClusterKinematics() {
            int  warp_idx     = threadIdx.x % CUDA_WARP_SIZE;
            bool leading_lane = !(threadIdx.x >> CUDA_WARP_SHIFT);

            FinalizeSharedMem* smem = (FinalizeSharedMem*)mcutil::cache_univ;
            __syncthreads();

            smem->dim_original        = Buffer::model_cached->current_field_size;
            smem->iter_original       = Buffer::model_cached->iter_1body;
            smem->elastic_count[0]    = 0;
            smem->elastic_count[1]    = 0;
            for (int ib1 = 0; ib1 <= Buffer::model_cached->n_cluster / CUDA_WARP_SIZE; ++ib1) {
                // initialize shared
                smem->cluster_z[warp_idx]      = 0;
                smem->cluster_a[warp_idx]      = 0;
                smem->cluster_mass[warp_idx]   = 0.f;
                smem->cluster_excit[warp_idx]  = 0.f;
                smem->cluster_u[warp_idx]      = 0.f;
                smem->cluster_v[warp_idx]      = 0.f;
                smem->cluster_w[warp_idx]      = 0.f;
                __syncthreads();

                // calculate cluster kinematics one by one
                for (int ic = 0; ic < CUDA_WARP_SIZE; ++ic) {
                    int target = ib1 * CUDA_WARP_SIZE + ic;
                    if (target >= Buffer::model_cached->n_cluster)
                        break;

                    for (int ib2 = 0; ib2 < smem->iter_original; ++ib2) {
                        int   znum = 0;
                        int   anum;
                        int   pid  = ib2 * blockDim.x + threadIdx.x;
                        if (pid < smem->dim_original) {
                            pid      += Buffer::model_cached->offset_1d;
                            int flags = Buffer::Participant::flags[pid];
                            if (flags >> CLUSTER_IDX_SHIFT == target &&
                                flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER) {
                                znum = flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON ? 1 : 0;
                                pid -= Buffer::model_cached->offset_1d;
                                anum = atomicAdd(&smem->cluster_a[ic], 1);
                                Buffer::model_cached->participant_idx[anum] = pid;
                            }
                        }
                        __syncthreads();
                        znum = rsum32_(znum);
                        if (!warp_idx)  // leading warp
                            atomicAdd(&smem->cluster_z[ic], znum);
                        __syncthreads();
                    }

                    // handle unphysical cluster
                    if (smem->cluster_a[ic] == 0 || smem->cluster_z[ic] == 0 || smem->cluster_z[ic] == smem->cluster_a[ic]) {
                        for (int ib2 = 0; ib2 < smem->iter_original; ++ib2) {
                            int pid = ib2 * blockDim.x + threadIdx.x;
                            if (pid < smem->dim_original) {
                                pid      += Buffer::model_cached->offset_1d;
                                int flags = Buffer::Participant::flags[pid];
                                if (flags >> CLUSTER_IDX_SHIFT == target &&
                                    flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER) {
                                    flags &= ~PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER;
                                    Buffer::Participant::flags[pid] = flags;
                                }
                            }
                            __syncthreads();
                        }
                        smem->cluster_a[ic] = 0;
                        smem->cluster_z[ic] = 0;
                    }
                    else {
                        // calculate internal energy
                        Buffer::model_cached->za_system.x = smem->cluster_z[ic];
                        Buffer::model_cached->za_system.y = smem->cluster_a[ic];
                        setDimension(smem->cluster_a[ic]);
                        __syncthreads();
                        cal2BodyQuantities();
                        calExcitationEnergyAndMomentum();
                        __syncthreads();
                        // write to shared memory
                        smem->cluster_mass[ic]  = Buffer::model_cached->mass_system;
                        smem->cluster_excit[ic] = Buffer::model_cached->excitation;
                        smem->cluster_u[ic]     = Buffer::model_cached->momentum_system[0];
                        smem->cluster_v[ic]     = Buffer::model_cached->momentum_system[1];
                        smem->cluster_w[ic]     = Buffer::model_cached->momentum_system[2];

                        // elastic like counter
                        if (!threadIdx.x) {
                            if (smem->cluster_z[ic] == Buffer::model_cached->za_nuc[0].x &&
                                smem->cluster_a[ic] == Buffer::model_cached->za_nuc[0].y &&
                                smem->cluster_excit[ic] < 1.e-3f && !smem->elastic_count[0]) {
                                smem->elastic_count[0] = 1;
                            }
                            else if (smem->cluster_z[ic] == Buffer::model_cached->za_nuc[1].x &&
                                     smem->cluster_a[ic] == Buffer::model_cached->za_nuc[1].y &&
                                     smem->cluster_excit[ic] < 1.e-3f && !smem->elastic_count[1]) {
                                     smem->elastic_count[1] = 1;
                            }
                        }
                        __syncthreads();
                    }
                    if (smem->elastic_count[0] && smem->elastic_count[1])  // elastic
                        return false;
                }
                __syncthreads();

                if (leading_lane) {
                    // Rotate azimuthally
                    float tempx = smem->cluster_u[warp_idx];
                    float tempy = smem->cluster_v[warp_idx];
                    smem->cluster_u[warp_idx] = tempx * Buffer::model_cached->cphi - tempy * Buffer::model_cached->sphi;
                    smem->cluster_v[warp_idx] = tempx * Buffer::model_cached->sphi + tempy * Buffer::model_cached->cphi;

                    // Lorentz boost to LAB system
                    float ln = Buffer::model_cached->beta_lab_nn;
                    float e  =
                        smem->cluster_mass[warp_idx] * smem->cluster_mass[warp_idx] +
                        smem->cluster_u[warp_idx]    * smem->cluster_u[warp_idx] +
                        smem->cluster_v[warp_idx]    * smem->cluster_v[warp_idx] +
                        smem->cluster_w[warp_idx]    * smem->cluster_w[warp_idx];
                    e = sqrtf(e);
                    float g  = 1.f / sqrtf(1.f - ln * ln);
                    float bp = smem->cluster_w[warp_idx] * ln;
                    smem->cluster_w[warp_idx] += ln * g * (g / (1.f + g) * bp - e);
                    e = g * (e - bp);

                    // momentum to direction vector
                    //float norm =
                    //    smem->cluster_u[warp_idx] * smem->cluster_u[warp_idx] +
                    //    smem->cluster_v[warp_idx] * smem->cluster_v[warp_idx] +
                    //    smem->cluster_w[warp_idx] * smem->cluster_w[warp_idx];
                    //norm = sqrtf(norm);
                    //smem->cluster_u[warp_idx] /= norm;
                    //smem->cluster_v[warp_idx] /= norm;
                    //smem->cluster_w[warp_idx] /= norm;

                    // Rotate to initial direction
                    float vs = 
                        Buffer::model_cached->initial_polar[1] * smem->cluster_u[warp_idx] +
                        Buffer::model_cached->initial_polar[0] * smem->cluster_w[warp_idx];
                    smem->cluster_w[warp_idx] = 
                        Buffer::model_cached->initial_polar[1] * smem->cluster_w[warp_idx] - 
                        Buffer::model_cached->initial_polar[0] * smem->cluster_u[warp_idx];
                    smem->cluster_u[warp_idx] =
                        Buffer::model_cached->initial_azim[1] * vs -
                        Buffer::model_cached->initial_azim[0] * smem->cluster_v[warp_idx];
                    smem->cluster_v[warp_idx] =
                        Buffer::model_cached->initial_azim[0] * vs +
                        Buffer::model_cached->initial_azim[1] * smem->cluster_v[warp_idx];

                    // za packing
                    mcutil::UNION_FLAGS deex_flags(Buffer::model_cached->initial_flags);
                    deex_flags.deex.z = smem->cluster_z[warp_idx];
                    deex_flags.deex.a = smem->cluster_a[warp_idx];

                    // push
                    if (smem->cluster_a[warp_idx] > 0) {  // do not write nonsense cluster
                        int idx = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].pushAtomic();
                        // position
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[idx] 
                            = Buffer::model_cached->initial_position[0];
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[idx]     
                            = Buffer::model_cached->initial_position[1];
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[idx]    
                            = Buffer::model_cached->initial_position[2];

                        // weight
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[idx]
                            = Buffer::model_cached->initial_weight;

                        // flag
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[idx]
                            = deex_flags.astype<unsigned int>();

                        // momentum
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[idx]     
                            = smem->cluster_u[warp_idx];
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[idx]     
                            = smem->cluster_v[warp_idx];
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[idx]    
                            = smem->cluster_w[warp_idx];

                        assertNAN(smem->cluster_u[warp_idx]);
                        assertNAN(smem->cluster_v[warp_idx]);
                        assertNAN(smem->cluster_w[warp_idx]);

                        // energy
                        buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].e[idx]
                            = smem->cluster_excit[warp_idx] * 1e3f;  // MeV

                        assertNAN(smem->cluster_excit[warp_idx]);

                        if (BUFFER_HAS_HID) {
                            buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[idx]
                                = Buffer::model_cached->initial_hid;
                        }
                    }
                }
                __syncthreads();

            }
            setDimension(smem->dim_original);
            return true;
        }


        __device__ void writeSingleNucleons() {
            for (int ib = 0; ib < Buffer::model_cached->iter_1body; ++ib) {
                int i = ib * blockDim.x + threadIdx.x;
                if (i < Buffer::model_cached->current_field_size) {
                    i += Buffer::model_cached->offset_1d;
                    int   flags = Buffer::Participant::flags[i];
                    float px    = Buffer::Participant::momentum_x[i];
                    float py    = Buffer::Participant::momentum_y[i];
                    float pz    = Buffer::Participant::momentum_z[i];
                    float mass  = Buffer::Participant::mass[i];
                    if (!(flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_IN_CLUSTER)) {
                        // rotate azimutally
                        float tempx = px;
                        float tempy = py;
                        px = tempx * Buffer::model_cached->cphi - tempy * Buffer::model_cached->sphi;
                        py = tempx * Buffer::model_cached->sphi + tempy * Buffer::model_cached->cphi;

                        // Lorentz boost to lab system
                        float ln = Buffer::model_cached->beta_lab_nn;
                        float e  = mass * mass + px * px + py * py + pz * pz;
                        e = sqrtf(e);
                        float g = 1.f / sqrtf(1.f - ln * ln);
                        float bp = pz * ln;
                        pz += ln * g * (g / (1.f + g) * bp - e);
                        e   = g * (e - bp);

                        // Rotate to initial direction
                        float vs = 
                            Buffer::model_cached->initial_polar[1] * px +
                            Buffer::model_cached->initial_polar[0] * pz;
                        pz =
                            Buffer::model_cached->initial_polar[1] * pz -
                            Buffer::model_cached->initial_polar[0] * px;
                        px =
                            Buffer::model_cached->initial_azim[1] * vs -
                            Buffer::model_cached->initial_azim[0] * py;
                        py =
                            Buffer::model_cached->initial_azim[0] * vs +
                            Buffer::model_cached->initial_azim[1] * py;

                        mcutil::UNION_FLAGS deex_flags(Buffer::model_cached->initial_flags);
                        deex_flags.deex.z = flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON ? 1 : 0;
                        deex_flags.deex.a = 1;

                        // kinetic energy [GeV]
                        e -= mass;

                        // target
                        bool target = flags & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

                        // push
                        int idx = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].pushAtomic();
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].x[idx]     
                            = Buffer::model_cached->initial_position[0];
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].y[idx]     
                            = Buffer::model_cached->initial_position[1];
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].z[idx]     
                            = Buffer::model_cached->initial_position[2];

                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].u[idx] = px;
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].v[idx] = py;
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].w[idx] = pz;

                        assertNAN(px);
                        assertNAN(py);
                        assertNAN(pz);
   
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].wee[idx]   
                            = Buffer::model_cached->initial_weight;
                        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].flags[idx]
                            = deex_flags.astype<unsigned int>();

                        if (BUFFER_HAS_HID) {
                            buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].hid[idx]
                                = Buffer::model_cached->initial_hid;
                        }
                    }
                }
                __syncthreads();
            }
        }


#ifndef NDEBUG


        __device__ int   __ty[__SYSTEM_TEST_DIMENSION] = {
            1 ,1 ,1, 1, 1, 1,
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0
        };
        __device__ float __rx[__SYSTEM_TEST_DIMENSION] = {
           2.53038 , -9.52212,   -11.1372,  0.219025,  2.82231, -16.9309,
           -8.90082,  12.6091,     -12.21,   2.52974,  2.94799,  3.40433,
           -1.93745, -1.15723,  0.0418667,   5.45261, -4.38807,  31.0403,
           28.5499 , -14.5001,   -2.66397,  -1.93346, -1.37424, -3.57328
        };
        __device__ float __ry[__SYSTEM_TEST_DIMENSION] = {
           -2.52852,	14.7567 ,	-0.399907,	24.7643 ,	-1.31646,	10.5735 ,
            -29.2832,	-8.59097,	8.42143 ,	2.82926 ,	-1.27115,	-1.50691,
            -1.96031,	0.537173,	1.62217 ,	0.187909,	-1.4294	,   -1.48736,
            0.597406,	-27.2868,	-0.825821,	0.132811,	-5.22105,	19.2131
        };
        __device__ float __rz[__SYSTEM_TEST_DIMENSION] = {
            -29.112,	-8.64642,	-10.2827,	-28.0772,	-27.8385,	-29.5485,
            -23.659,	11.6198 ,	-41.1155,	-29.9457,	-29.8894,	-29.1576,
            32.5782,	31.4271 ,	31.808  ,	16.9297 ,	31.34   ,	19.1916 ,
            14.3593,	-17.9297,	29.2733 ,	33.3002 ,	31.1115 ,	27.3467
        };
        __device__ float __px[__SYSTEM_TEST_DIMENSION] = {
            0.178502 ,	-0.100223 ,	-0.135871,	0.000617116,	0.0185447 ,	-0.194152,
            -0.11589 ,	0.12234   ,	-0.145786,	0.0597584  ,	-0.0830832,	0.0825837,
            0.0191552,	-0.0626398,	-0.013422,	0.0654593  ,	-0.0843839,	0.348197 ,
            0.341473 ,	-0.16333  ,	0.010514 ,	0.00853691 ,	-0.0872362,	-0.055474
        };
        __device__ float __py[__SYSTEM_TEST_DIMENSION] = {
            -0.00371677,	0.180591 ,	-0.00667258,	0.269705  ,	-0.00553188,	0.140325  ,
            -0.332989  ,	-0.104186,	0.0993201  ,	-0.0150453,	-0.00719173,	-0.0252185,
            -0.00857555,	0.0137058,	-0.0728963 ,	0.00372519,	0.0648226  ,	-0.0226284,
            0.0338322  ,	-0.337268,	-0.0916963 ,	0.018588  ,	0.00110179 ,	0.208724
        };
        __device__ float __pz[__SYSTEM_TEST_DIMENSION] = {
            -0.292523,	-0.0936555,	-0.107785,	-0.310559,	-0.3167  ,	-0.381225,
            -0.27309 ,	0.13309   ,	-0.496655,	-0.345646,	-0.344711,	-0.339696,
            0.361957 ,	0.433021  ,	0.365314 ,	0.206358 ,	0.303008 ,	0.222506 ,
            0.201168 ,	-0.22139  ,	0.354203 ,	0.276705 ,	0.350121 ,	0.317477
        };


        __device__ void testG4QMDSampleSystem() {
            
            for (int iter = 0; iter <= __SYSTEM_TEST_DIMENSION / blockDim.x; ++iter) {
                int pid = iter * blockDim.x + threadIdx.x;
                if (pid < __SYSTEM_TEST_DIMENSION) {
                    Buffer::model_cached->participant_idx[pid] = (unsigned char)pid;
                    int flag = 0;
                    int mi   = Buffer::model_cached->offset_1d + pid;
                    flag |= (pid < __TEST_PROJECTILE_ZA.y)
                        ? PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE
                        : PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET;
                    if (__ty[pid])
                        flag |= PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    Buffer::Participant::flags[mi] = flag;
                    Buffer::Participant::mass[mi] = __ty[pid]
                        ? ::constants::MASS_PROTON_GEV
                        : ::constants::MASS_NEUTRON_GEV;
                    Buffer::Participant::position_x[mi] = __rx[pid];
                    Buffer::Participant::position_y[mi] = __ry[pid];
                    Buffer::Participant::position_z[mi] = __rz[pid];
                    Buffer::Participant::momentum_x[mi] = __px[pid];
                    Buffer::Participant::momentum_y[mi] = __py[pid];
                    Buffer::Participant::momentum_z[mi] = __pz[pid];
                }
                __syncthreads();
            }           
            setDimension(__SYSTEM_TEST_DIMENSION);
            //cal2BodyQuantities();
            //calGraduate();
            
            //for (int i = 0; i < constants::PROPAGATE_MAX_TIME; ++i) {
            //    doPropagate();
            //    Collision::calKinematicsOfBinaryCollisions();
            //    MeanField::calTotalKineticEnergy();
            //    MeanField::calTotalPotentialEnergy();
            //    QMD_DUMP_ACTION__;
            //}
        }


#endif


    }
}