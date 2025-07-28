
#include "collision.cuh"
#include "xs.cuh"
#include "mean_field.cuh"

#include "device/shuffle.cuh"

#include <stdio.h>


namespace RT2QMD {
    namespace Collision {


        __constant__ bool USING_INCL_NN_SCATTERING;
        __device__   Hadron::NNScatteringTable* g4_nn_table;


        __host__ cudaError_t setNNTable(CUdeviceptr table_handle, bool use_incl_model) {
            M_SOAPtrMapper(bool, use_incl_model, USING_INCL_NN_SCATTERING);
            if (!use_incl_model) {
                M_SOASymbolMapper(Hadron::NNScatteringTable*, table_handle, g4_nn_table);
            }
            return cudaSuccess;
        }


        __device__ void sortCollisionCandidates() {
            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;
            __syncthreads();

            int n_bc      = smem->n_binary_candidate;
            int n_bc_ceil = max(CUDA_WARP_SIZE, pow2_ceil_(n_bc));
            __syncthreads();

            unsigned short* can_sptr = reinterpret_cast<unsigned short*>(smem->binary_candidate);

            // bitonic sort
            for (int stride_max = CUDA_WARP_SIZE; stride_max <= n_bc_ceil; stride_max <<= 1) {
                // block level
                for (int stride = stride_max / 2; stride >= CUDA_WARP_SIZE; stride >>= 1) {
                    for (int i = 0; i < (n_bc + blockDim.x - 1) / blockDim.x; ++i) {
                        int bc_idx   = i * blockDim.x + threadIdx.x;
                        int pair_idx = bc_idx ^ stride;

                        if (bc_idx < pair_idx) {
                            unsigned short a = can_sptr[bc_idx];
                            unsigned short b = can_sptr[pair_idx];

                            bool ascending = ((bc_idx / stride_max) % 2 == 0);
                            if ((a > b) == ascending) {
                                can_sptr[bc_idx]   = b;
                                can_sptr[pair_idx] = a;
                            }
                        }
                        __syncthreads();
                    }
                }
                __syncthreads();
                // warp level
                for (int i = 0; i < (n_bc + blockDim.x - 1) / blockDim.x; ++i) {
                    int bc_idx  = i * blockDim.x + threadIdx.x;
                    int unf_can = can_sptr[bc_idx];
                    if (bc_idx >= n_bc && stride_max == CUDA_WARP_SIZE)
                        unf_can = INT32_MAX;
                    __syncthreads();
                    // sort (zigzag)
                    unf_can = bsort32_(unf_can);

                    int lane_id = threadIdx.x % CUDA_WARP_SIZE;
                    if (((bc_idx / stride_max) & 1) != 0)
                        unf_can = __shfl_sync(0xffffffff, unf_can, CUDA_WARP_SIZE - 1 - lane_id);
                    // back to shared
                    can_sptr[bc_idx] = (unsigned short)unf_can;
                    __syncthreads();
                }
            }
        }


        __device__ void elasticScatteringNN() {
            int   lane_idx     = threadIdx.x  % CUDA_WARP_SIZE;
            int   warp_idx     = threadIdx.x >> CUDA_WARP_SHIFT;
            int   lane_leading = lane_idx / 2 * 2;
            float temp;

            if (warp_idx)
                return;

            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;

            bool isospin = smem->flag[lane_leading] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON
                == smem->flag[lane_leading + 1] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

            // build boost vector
            float ene   = 0.f;
                  temp  = 0.f;
            float beta2 = 0.f;
            float mass  = smem->mass[lane_idx];
            ene    = mass * mass;
            temp   = smem->px[lane_idx];
            ene   += temp * temp;
            temp  += __shfl_xor_sync(0xffffffff, temp, 1);
            smem->boostx[lane_idx] = temp;
            beta2 += temp * temp;
            temp   = smem->py[lane_idx];
            ene   += temp * temp;
            temp  += __shfl_xor_sync(0xffffffff, temp, 1);
            smem->boosty[lane_idx] = temp;
            beta2 += temp * temp;
            temp   = smem->pz[lane_idx];
            ene   += temp * temp;
            temp  += __shfl_xor_sync(0xffffffff, temp, 1);
            smem->boostz[lane_idx] = temp;
            beta2 += temp * temp;
            ene    = sqrtf(ene);
            temp   = ene;
            temp  += __shfl_xor_sync(0xffffffff, temp,  1);
            beta2 /= temp * temp;

            smem->boostx[lane_idx] /= temp;  // boost vector LAB -> CM
            smem->boosty[lane_idx] /= temp;
            smem->boostz[lane_idx] /= temp;
            __syncwarp();

            // boost to CM
            float gamma = 1.f / sqrtf(1.f - beta2);
            float bp    =
                smem->px[lane_idx] * smem->boostx[lane_idx] +
                smem->py[lane_idx] * smem->boosty[lane_idx] +
                smem->pz[lane_idx] * smem->boostz[lane_idx];  // LAB -> CM

            temp = gamma * (gamma / (1.f + gamma) * bp - ene);

            smem->px[lane_idx] += smem->boostx[lane_idx] * temp;
            smem->py[lane_idx] += smem->boosty[lane_idx] * temp;
            smem->pz[lane_idx] += smem->boostz[lane_idx] * temp;

            ene += __shfl_xor_sync(0xffffffff, ene, 1);

            float e2cm = (1.f - beta2) * ene * ene;  // square of total energy of system in CM [GeV^2]

            // lab momentum
            mass *= mass;  // Mass^2
            float mt = mass;
            float md = mass;
            float m2 = __shfl_xor_sync(0xffffffff, mt, 1);
            mt += m2;
            md -= m2;
            m2  = lane_idx % 2 ? m2 : mass;
            float plab = e2cm * e2cm - 2.f * e2cm * mt + md * md;

            plab /= 4.f * m2;
            plab  = sqrtf(plab);

            float ctet;  // cosine theta

            float xx =
                smem->px[lane_leading] * smem->px[lane_leading] +
                smem->py[lane_leading] * smem->py[lane_leading];  // 1E-6
            float zz =
                smem->pz[lane_leading] * smem->pz[lane_leading];  // 1E-6

            float psq   = xx + zz;     // 1E-6
            float pnorm = sqrtf(psq);  // 1E-3

            if (USING_INCL_NN_SCATTERING) {  // INCL style
                float b     = Hadron::calculateNNAngularSlope(plab, isospin);  // 1E+6
                float btmax = 4.f * psq * b;  // 1
                float z     = expf(-btmax);   // 1
                      temp  = curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]);
                float y     = 1.f - temp * (1.f - z);   // 1
                float t     = logf(y) / b;              // 1E-6

                int   iexpi = 0;
                float apt   = 1.f;

                // Handle np case
                if (!isospin) {
                    if (plab > 0.8f) {
                        apt          = 0.8f / plab;  // 1
                        apt         *= apt;  // 1
                        float cpt    = fmaxf(6.23f * expf(-1.79f * plab), 0.3f);  // 1
                        float aaa    = (1.f + apt) * (1.f - expf(-btmax)) / b;    // 1E-6
                        float argu   = psq * 1e+2f;  // 1
                        argu         = (argu >= 8.f) ? 0.f : expf(-4.f * argu);  // 1

                        float aac    = cpt * (1.f - argu) * 1e-2f;  // 1E-6
                        float fracpn = aaa / (aac + aaa);
                        temp         = curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]);
                        if (temp > fracpn) {
                            z     = expf(-4.f * psq * 1e+2f);  // 1
                            iexpi = 1.f;
                            temp  = curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]);
                            y     = 1.f - temp * (1.f - z);
                            t     = logf(y) * 1e-2f;
                        }
                    }
                }
                ctet = 1.f + 0.5f * t / psq;  // 1
            }
            else {  // Geant4 style NN
                float elab = smem->mass[lane_leading];
                elab = sqrtf(plab * plab + elab * elab) - elab;

                int   je1    = 0;
                int   je2    = g4_nn_table->nenergy[isospin] - 1;
                int   nangle = g4_nn_table->nangle[isospin];
                do {
                    int mid_bin = (je1 + je2) / 2;
                    if (elab < g4_nn_table->elab[isospin][mid_bin])
                        je2 = mid_bin;
                    else
                        je1 = mid_bin;
                } while (je2 - je1 > 1);
                __syncwarp();
                float delab = g4_nn_table->elab[isospin][je2] - g4_nn_table->elab[isospin][je1];

                float b     = g4_nn_table->sig[isospin][je1 * nangle];
                float dsig  = g4_nn_table->sig[isospin][je2 * nangle] - b;
                float rc    = dsig / delab;
                b -= rc * g4_nn_table->elab[isospin][je1];
                
                float sigint1 = rc * elab + b;
                float sigint2 = 0;

                // sample the direction cosine
                float sample = curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]);
                int   ke1    = 0;
                int   ke2    = nangle - 1;
                do {
                    int mid_bin = (ke1 + ke2) / 2;
                    b    = g4_nn_table->sig[isospin][je1 * nangle + mid_bin];
                    dsig = g4_nn_table->sig[isospin][je2 * nangle + mid_bin] - b;
                    rc   = dsig / delab;
                    b -= rc * g4_nn_table->elab[isospin][je1];

                    float sigint = rc * elab + b;
                    if (sample < sigint) {
                        ke2     = mid_bin;
                        sigint2 = sigint;
                    }
                    else {
                        ke1     = mid_bin;
                        sigint1 = sigint;
                    }
                } while (ke2 - ke1 > 1);
                __syncwarp();

                dsig = sigint2 - sigint1;
                rc   = 1.f / dsig;
                b    = (float)ke1 - rc * sigint1;
                ctet = cosf((0.5 + rc * sample + b) * ::constants::FP32_DEG_TO_RAD);
            }
            ctet = fmaxf(-1.f, fminf(1.f, ctet));

            float stet = sqrtf(fmaxf(0.f, 1.f - ctet * ctet));  // 1
            float fi   = ::constants::FP32_TWO_PI * curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]);
            float cfi, sfi;
            __sincosf(fi, &sfi, &cfi);

            float3 pnew;
            if (xx >= zz * 1e-8f) {
                float yn = sqrtf(xx);   // 1E-3
                float zn = yn * pnorm;  // 1E-6

                pnew.x = smem->px[lane_leading] * ctet + stet * pnorm * (
                    smem->py[lane_leading] / yn * cfi +
                    smem->px[lane_leading] * smem->pz[lane_leading] / zn * sfi
                    );
                pnew.y = smem->py[lane_leading] * ctet + stet * pnorm * (
                    -smem->px[lane_leading] / yn * cfi +
                    smem->py[lane_leading] * smem->pz[lane_leading] / zn * sfi
                    );
                pnew.z = smem->pz[lane_leading] * ctet + stet * pnorm * (
                    -xx / zn * sfi
                    );
            }
            else {
                pnew.x = smem->pz[lane_leading] * cfi * stet;
                pnew.y = smem->pz[lane_leading] * sfi * stet;
                pnew.z = smem->pz[lane_leading] * ctet;
            }
            __syncwarp();

            // two particles should be opposite after collision
            float grad = 0.f;
            temp   = smem->rx[lane_leading + 1] - smem->rx[lane_leading];
            temp  *= pnew.x;
            grad  += temp;
            temp   = smem->ry[lane_leading + 1] - smem->ry[lane_leading];
            temp  *= pnew.y;
            grad  += temp;
            temp   = smem->rz[lane_leading + 1] - smem->rz[lane_leading];
            temp  *= pnew.z;
            grad  += temp;
            if (grad >= 0.f) {
                pnew.x = -pnew.x;
                pnew.y = -pnew.y;
                pnew.z = -pnew.z;
            }
            __syncwarp();

            // backscattering
            if (lane_idx % 2) {
                if (curand_uniform(&rand_state[blockIdx.x * blockDim.x + threadIdx.x]) > 0.5f) {
                    int temp = smem->flag[lane_leading];
                    smem->flag[lane_leading]     = smem->flag[lane_leading + 1];
                    smem->flag[lane_leading + 1] = temp;
                }
            }
            __syncwarp();

            temp   = __shfl_xor_sync(0xffffffff, pnew.x, 1);
            pnew.x = lane_idx % 2 ? -temp : pnew.x;
            temp   = __shfl_xor_sync(0xffffffff, pnew.y, 1);
            pnew.y = lane_idx % 2 ? -temp : pnew.y;
            temp   = __shfl_xor_sync(0xffffffff, pnew.z, 1);
            pnew.z = lane_idx % 2 ? -temp : pnew.z;

            // boost to lab system
            ene   = smem->mass[lane_idx] * smem->mass[lane_idx];
            ene  += pnew.x * pnew.x;
            ene  += pnew.y * pnew.y;
            ene  += pnew.z * pnew.z;
            ene   = sqrtf(ene);
            beta2 =
                smem->boostx[lane_idx] * smem->boostx[lane_idx] +
                smem->boosty[lane_idx] * smem->boosty[lane_idx] +
                smem->boostz[lane_idx] * smem->boostz[lane_idx];
            gamma = 1.f / sqrtf(1.f - beta2);
            bp    =
                pnew.x * -smem->boostx[lane_idx] +
                pnew.y * -smem->boosty[lane_idx] +
                pnew.z * -smem->boostz[lane_idx];  // CM -> LAB
            temp = gamma * (gamma / (1.f + gamma) * bp - ene);
            pnew.x -= smem->boostx[lane_idx] * temp;
            pnew.y -= smem->boosty[lane_idx] * temp;
            pnew.z -= smem->boostz[lane_idx] * temp;

            smem->px[lane_idx] = pnew.x;
            smem->py[lane_idx] = pnew.y;
            smem->pz[lane_idx] = pnew.z;
            __syncwarp();
            return;
        }


        __device__ void calPotentialSegBeforeBinaryCollision(int prid, int ip) {
            int   lane_idx = threadIdx.x  % CUDA_WARP_SIZE;
            int   warp_idx = threadIdx.x >> CUDA_WARP_SHIFT;

            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;

            ip   += warp_idx % 2;
            int  i         = (warp_idx % 2) ? smem->binary_candidate[prid].y : smem->binary_candidate[prid].x;
            bool is_proton = smem->flag[ip] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
            __syncthreads();

            float pot = MeanField::getPotential(i, is_proton);
            if (warp_idx < 2 && !lane_idx)
                smem->pot_ini[warp_idx] = pot;
            return;
        }


        __device__ void calPotentialSegAfterBinaryCollision(int prid, int ip) {
            int   lane_idx = threadIdx.x % CUDA_WARP_SIZE;
            int   warp_idx = threadIdx.x >> CUDA_WARP_SHIFT;
            
            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;

            ip   += warp_idx % 2;
            uchar2 bcti;
            bcti.x  = (warp_idx % 2) ? smem->binary_candidate[prid].y : smem->binary_candidate[prid].x;
            bcti.y  = (warp_idx % 2) ? smem->binary_candidate[prid].x : smem->binary_candidate[prid].y;
            bool ci = smem->flag[ip] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
            bool cj;

            float rhoa = 0.f;
            float rhos = 0.f;
            float rhoc = 0.f;
            float rho3;

            float3 pj;  // opposite
            float  temp;
            for (int jw = 0; jw < Buffer::model_cached->iter_gwarp; ++jw) {
                int j = lane_idx + jw * CUDA_WARP_SIZE;
                int pidj;
                if (j < Buffer::model_cached->current_field_size) {
                    pidj = (int)Buffer::model_cached->participant_idx[j] + Buffer::model_cached->offset_1d;
                    cj   = Buffer::Participant::flags[pidj] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

                    // momentum
                    pj.x = Buffer::Participant::momentum_x[pidj];
                    pj.y = Buffer::Participant::momentum_y[pidj];
                    pj.z = Buffer::Participant::momentum_z[pidj];

                    if (bcti.x == j || bcti.y == j) {
                        pj.x = smem->px[ip ^ (bcti.y == j)];
                        pj.y = smem->py[ip ^ (bcti.y == j)];
                        pj.z = smem->pz[ip ^ (bcti.y == j)];
                    }

                    // energy
                    float mj = Buffer::Participant::mass[pidj];

                    float ei = sqrtf(
                        smem->mass[ip] * smem->mass[ip] + 
                        smem->px[ip]   * smem->px[ip]   +
                        smem->py[ip]   * smem->py[ip]   + 
                        smem->pz[ip]   * smem->pz[ip]
                    );
                    float ej = sqrtf(mj * mj + pj.x * pj.x + pj.y * pj.y + pj.z * pj.z);

                    float rbrb = 0.f;
                    float rij2 = 0.f;
                    float pij2 = 0.f;
                    float gij2 = 0.f;

                    float pidr = 0.f;
                    float pjdr = 0.f;

                    // vector x
                    float rij;
                    rij   = smem->rx[ip];
                    rij  -= Buffer::Participant::position_x[pidj];  // dr.x (static)
                    assertNAN(rij);
                    pidr += rij * smem->px[ip];
                    pjdr += rij * pj.x;
                    temp  = smem->px[ip] + pj.x;   // p4ij.x
                    rbrb += rij  * temp;           // rij.x  * p4ij.x
                    gij2 += temp * temp;           // p4ij.x * p4ij.x
                    rij2 += rij  * rij;            // rij.x  * rij.x  &&  rsq
                    temp  = smem->px[ip] - pj.x;   // pij.x
                    pij2 += temp * temp;           // pij.x  * pij.x

                    // vector y
                    rij   = smem->ry[ip];
                    rij  -= Buffer::Participant::position_y[pidj];
                    assertNAN(rij);
                    pidr += rij * smem->py[ip];
                    pjdr += rij * pj.y;
                    temp  = smem->py[ip] + pj.y;
                    rbrb += rij * temp;
                    gij2 += temp * temp;
                    rij2 += rij * rij;
                    temp  = smem->py[ip] - pj.y;
                    pij2 += temp * temp;

                    // distance vector z
                    rij   = smem->rz[ip];
                    rij  -= Buffer::Participant::position_z[pidj];
                    assertNAN(rij);
                    pidr += rij * smem->pz[ip];
                    pjdr += rij * pj.z;
                    temp  = smem->pz[ip] + pj.z;
                    rbrb += rij  * temp;
                    gij2 += temp * temp;
                    rij2 += rij  * rij;
                    temp  = smem->pz[ip] - pj.z;
                    pij2 += temp * temp;

                    temp = ei + ej;
                    rbrb /= temp;         // rij * p4ij.boost = (rij.x * p4ij.x + rij.y * p4ij.y + rij.z * p4ij.z) / p4ij.e
                    gij2 /= temp * temp;  // p4ij.boost.mag2
                    gij2 = 1.f / sqrtf(1.f - gij2); // gammaij
                    gij2 = gij2 * gij2;
                    rij2 += gij2 * rbrb * rbrb;

                    // mean field matrix element
                    // Gauss term
                    temp  = -rij2 * constants::D_SKYRME_C0;

                    float rha;
                    rha   = temp > constants::PAULI_EPSX ? expf(temp) : 0.f;
                    rha   = bcti.x == j ? 0.f : rha;

                    rhoa += rha;

                    // Coulomb term
                    rij2 += constants::COULOMB_EPSILON;
                    temp  = sqrtf(rij2);  // rrs
                    temp  = temp * constants::D_SKYRME_C0_S < 5.8
                        ? erff(temp * constants::D_SKYRME_C0_S) / temp
                        : 1.f / temp;  // erfij
                    temp  = bcti.x == j ? 0.f : temp;
                    temp  = (float)(ci * cj) * temp;

                    rhoc += temp;

                    // symmetry term
                    rha   = ci == cj ? rha : -rha;
                    rhos += rha;
                }
                __syncthreads();
            }
            rhoa = rsum32_(rhoa);
            rhoc = rsum32_(rhoc);
            rhos = rsum32_(rhos);
            rho3 = powf(rhoa, constants::H_SKYRME_GAMM);
            
            if (warp_idx < 2 && !lane_idx) {
                smem->pot_fin[warp_idx] =
                    constants::H_SKYRME_C0 * rhoa +
                    constants::H_SKYRME_C3 * rho3 +
                    constants::H_SYMMETRY  * rhos +
                    constants::H_COULOMB   * rhoc;
            }
        }


        __device__ void calKinematicsOfBinaryCollisions() {
            int lane_idx = threadIdx.x  % CUDA_WARP_SIZE;
            int warp_idx = threadIdx.x >> CUDA_WARP_SHIFT;
            int lane_dim = blockDim.x  >> CUDA_WARP_SHIFT;
            bool exit_condition;

            __syncthreads();

            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;
            __syncthreads();

            if (!threadIdx.x) {
                exit_condition = !smem->n_binary_candidate;
                smem->exit_condition = exit_condition;
            }
            __syncthreads();
            exit_condition = smem->exit_condition;
            __syncthreads();
            //if (exit_condition)
            //    return;
            __syncthreads();

            // initialize hit mask
            if (threadIdx.x < MAX_DIMENSION_CLUSTER_B)
                smem->is_hit[threadIdx.x] = 0x0u;
            if (!threadIdx.x)
                smem->n_binary_candidate = min(smem->n_binary_candidate, BINARY_CANDIDATE_MAX);
            __syncthreads();

            // sort binary candidates
            // sortCollisionCandidates();

            // load data
            int n_bc = smem->n_binary_candidate;
            //if (threadIdx.x == 0 && blockIdx.x == 0) {
            //    for (int i = 0; i < n_bc; ++i) {
            //        printf("%d %d %d \n", blockIdx.x, smem->binary_candidate[i].x, smem->binary_candidate[i].y);
            //    }
            //}

            __syncthreads();
            for (int ib = 0; ib <= n_bc * BINARY_MAX_TRIAL_2 / CUDA_WARP_SIZE; ++ib) {
                // load global phase space for participant i
                for (int ip = 0; ip < Buffer::model_cached->iter_gphase; ++ip) {
                    int phid = ip * lane_dim + warp_idx;
                    int i    = (ib * CUDA_WARP_SIZE + lane_idx) / (BINARY_MAX_TRIAL * 2) * 2 + lane_idx % 2;
                    if (i < n_bc * 2 && phid < Buffer::Participant::SOA_MEMBERS) {
                        int pid = (int)((unsigned char*)(&smem->binary_candidate[0].x)[i]) + Buffer::model_cached->offset_1d;
                        mcutil::cache_univ[COLLISION_SHARED_MEM_OFFSET + CUDA_WARP_SIZE * phid + lane_idx]
                            = Buffer::Participant::ps[phid][pid];
                    }
                    __syncthreads();
                }
                assert(!(smem->mass[lane_idx] <= 0.0 && lane_idx < n_bc * 2));
                
                
                /*
                if (!warp_idx) {
                    int i = ib * CUDA_WARP_SIZE + lane_idx;
                    if (i < n_bc * 2) {
                        int pid = (int)((unsigned char*)(&smem->binary_candidate[0].x)[i]) + Buffer::model_cached->offset_1d;
                        smem->flag[lane_idx] = Buffer::Participant::flags[pid];
                        smem->mass[lane_idx] = Buffer::Participant::mass[pid];
                        smem->rx[lane_idx]   = Buffer::Participant::position_x[pid];
                        smem->ry[lane_idx]   = Buffer::Participant::position_y[pid];
                        smem->rz[lane_idx]   = Buffer::Participant::position_z[pid];
                        smem->px[lane_idx]   = Buffer::Participant::momentum_x[pid];
                        smem->py[lane_idx]   = Buffer::Participant::momentum_y[pid];
                        smem->pz[lane_idx]   = Buffer::Participant::momentum_z[pid];
                    }
                }
                __syncthreads();
                */

                elasticScatteringNN();
                __syncthreads();

                // violation of energy convervation & pauli blocking
                bool exit_condition;
                for (int ip = 0; ip < CUDA_WARP_SIZE; ip += 2) {
                    __syncthreads();
                    int phid = (ip + ib * CUDA_WARP_SIZE) / 2;  // number (multiplied by trial number)
                    int prid = phid >> BINARY_MAX_TRIAL_SHIFT;
                    if (prid >= n_bc)
                        break;

                    // first hit
                    __syncthreads();
                    if (!threadIdx.x) {
                        int pid;
                        pid = smem->binary_candidate[prid].x;
                        bool hit0 = smem->is_hit[pid >> 5] & (0x1u << (pid % 32));
                        pid = smem->binary_candidate[prid].y;
                        bool hit1 = smem->is_hit[pid >> 5] & (0x1u << (pid % 32));
                        smem->exit_condition = hit0 || hit1;
                    }
                    __syncthreads();
                    exit_condition = smem->exit_condition;
                    __syncthreads();
                    if (exit_condition) {
                        ip = ((ip / BINARY_MAX_TRIAL_2) + 1) * BINARY_MAX_TRIAL_2;
                        continue;
                    }
                    __syncthreads();

                    // calculate potential energy segment for participant i and j before collision (if first trial)
                    if (!(ip % BINARY_MAX_TRIAL_2))
                        calPotentialSegBeforeBinaryCollision(prid, ip);
                    __syncthreads();

                    // calculate potential energy after collision (all trial)
                    calPotentialSegAfterBinaryCollision(prid, ip);
                    __syncthreads();

                    // energetically forbidden collision
                    if (!threadIdx.x) {
                        smem->exit_condition = false;
                        if (fabsf(smem->pot_fin[0] + smem->pot_fin[1]
                            - smem->pot_ini[0] - smem->pot_ini[1]) > ENERGY_CONSERVATION_VIOLATION_THRES)
                            smem->exit_condition = true;
                    }
                    __syncthreads();
                    exit_condition = smem->exit_condition;
                    __syncthreads();
                    if (exit_condition)
                        continue;

                    // pauli block (skip all -> G4 RQMD)
                    if (isPauliBlocked(phid)) {
                        ip = ((ip / BINARY_MAX_TRIAL_2) + 1) * BINARY_MAX_TRIAL_2;
                        continue;
                    }

                    // accepted!

                    // hit mask
                    if (!threadIdx.x) {
                        int pid;
                        pid = smem->binary_candidate[prid].x;
                        smem->is_hit[pid >> 5] |= (0x1u << (pid % 32));
                        pid = smem->binary_candidate[prid].y;
                        smem->is_hit[pid >> 5] |= (0x1u << (pid % 32));
                    }
                    __syncthreads();
                    if (!(threadIdx.x / 2)) {
                        int pid        = lane_idx % 2 ? (int)smem->binary_candidate[prid].y : (int)smem->binary_candidate[prid].x;
                        pid += Buffer::model_cached->offset_1d;

                        int flag = smem->flag[ip + lane_idx % 2];
                        flag &= ~PARTICIPANT_FLAGS::PARTICIPANT_IS_PROJECTILE;
                        flag &= ~PARTICIPANT_FLAGS::PARTICIPANT_IS_TARGET;

                        assert(!isnan(smem->px[ip + lane_idx % 2]));
                        assert(!isnan(smem->py[ip + lane_idx % 2]));
                        assert(!isnan(smem->pz[ip + lane_idx % 2]));
                        /*
                        if (blockIdx.x < 10 && threadIdx.x == 1)
                        printf("%d %f %f %f %f %f %f \n",
                            pid,
                            Buffer::Participant::momentum_x[pid],
                            Buffer::Participant::momentum_y[pid],
                            Buffer::Participant::momentum_z[pid],
                            smem->px[ip + lane_idx % 2],
                            smem->py[ip + lane_idx % 2],
                            smem->pz[ip + lane_idx % 2]
                        );
                        */
                        Buffer::Participant::momentum_x[pid] = smem->px[ip + lane_idx % 2];
                        Buffer::Participant::momentum_y[pid] = smem->py[ip + lane_idx % 2];
                        Buffer::Participant::momentum_z[pid] = smem->pz[ip + lane_idx % 2];
                        Buffer::Participant::flags[pid] = flag;
                    }
                    __syncthreads();
                    if (!threadIdx.x)
                        Buffer::model_cached->n_collisions++;
                    ip = ((ip / BINARY_MAX_TRIAL_2) + 1) * BINARY_MAX_TRIAL_2;  // skip all broadcasted candidates
                }
                __syncthreads();
            }

            MeanField::cal2BodyQuantities();
            return;
        }


        __device__ bool isPauliBlocked(int candidate_id) {
            int lane_idx = threadIdx.x %  CUDA_WARP_SIZE;
            int warp_idx = threadIdx.x >> CUDA_WARP_SHIFT;
            int smid     = (candidate_id % (CUDA_WARP_SIZE / 2)) * 2;
            // shared memory
            CollisionSharedMem* smem = (CollisionSharedMem*)mcutil::cache_univ;
            
            float temp;
            float ei1;
            float ei2;
            bool  ci1 = smem->flag[smid]     & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
            bool  ci2 = smem->flag[smid + 1] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;

            // blocking p1
            ei1  = smem->mass[smid] * smem->mass[smid];
            ei1 += smem->px[smid]   * smem->px[smid];
            ei1 += smem->py[smid]   * smem->py[smid];
            ei1 += smem->pz[smid]   * smem->pz[smid];
            ei1  = sqrtf(ei1);

            // blocking p2
            ei2  = smem->mass[smid + 1] * smem->mass[smid + 1];
            ei2 += smem->px[smid + 1] * smem->px[smid + 1];
            ei2 += smem->py[smid + 1] * smem->py[smid + 1];
            ei2 += smem->pz[smid + 1] * smem->pz[smid + 1];
            ei2  = sqrtf(ei2);

            // 1st step reduction
            float pf1 = 0.f;
            float pf2 = 0.f;
            smem->boostx[lane_idx] = 0.f;  // using boostx for reduction 1
            smem->boosty[lane_idx] = 0.f;  // using boostx for reduction 2
            smem->pauli_blocked    = 0;
            for (int i = 0; i < Buffer::model_cached->iter_1body; ++i) {
                int   pid = threadIdx.x;
                if (pid < Buffer::model_cached->current_field_size) {
                    int   mai1     = (int)smem->binary_candidate[candidate_id].y 
                        * Buffer::model_cached->current_field_size + pid;
                    int   mai2     = (int)smem->binary_candidate[candidate_id].x
                        * Buffer::model_cached->current_field_size + pid;
                    float pij21    = 0.f;
                    float gij21    = 0.f;
                    float pij22    = 0.f;
                    float gij22    = 0.f;
                    float pj;
                    float ej;
                    bool  cj;
                    bool  smem_op1 = pid == (int)smem->binary_candidate[candidate_id].x;
                    bool  smem_op2 = pid == (int)smem->binary_candidate[candidate_id].y;

                    pid += Buffer::model_cached->offset_1d;
                    cj   = Buffer::Participant::flags[pid] & PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON;
                    ej   = Buffer::Participant::mass[pid];
                    ej  *= ej;

                    // vector x;
                    pj     = Buffer::Participant::momentum_x[pid];
                    pj     = smem_op1 ? smem->px[smid + 1] : pj;
                    pj     = smem_op2 ? smem->px[smid]     : pj;
                    ej    += pj * pj;
                    temp   = smem->px[smid] + pj;
                    gij21 += temp * temp;
                    temp   = smem->px[smid + 1] + pj;
                    gij22 += temp * temp;
                    temp   = smem->px[smid] - pj;
                    pij21 += temp * temp;
                    temp   = smem->px[smid + 1] - pj;
                    pij22 += temp * temp;

                    // vector y;
                    pj     = Buffer::Participant::momentum_y[pid];
                    pj     = smem_op1 ? smem->py[smid + 1] : pj;
                    pj     = smem_op2 ? smem->py[smid]     : pj;
                    ej    += pj * pj;
                    temp   = smem->py[smid] + pj;
                    gij21 += temp * temp;
                    temp   = smem->py[smid + 1] + pj;
                    gij22 += temp * temp;
                    temp   = smem->py[smid] - pj;
                    pij21 += temp * temp;
                    temp   = smem->py[smid + 1] - pj;
                    pij22 += temp * temp;

                    // vector z;
                    pj     = Buffer::Participant::momentum_z[pid];
                    pj     = smem_op1 ? smem->pz[smid + 1] : pj;
                    pj     = smem_op2 ? smem->pz[smid]     : pj;
                    ej    += pj * pj;
                    temp   = smem->pz[smid] + pj;
                    gij21 += temp * temp;
                    temp   = smem->pz[smid + 1] + pj;
                    gij22 += temp * temp;
                    temp   = smem->pz[smid] - pj;
                    pij21 += temp * temp;
                    temp   = smem->pz[smid + 1] - pj;
                    pij22 += temp * temp;

                    ej     = sqrtf(ej);

                    temp   = ei1 + ej;
                    gij21 /= temp * temp;
                    temp   = ei2 + ej;
                    gij22 /= temp * temp;

                    // 1st
                    temp = ci1 == cj ? 0.f : constants::MASS_DIF_2 / (ei1 + ej);
                    temp = temp * temp;
                    temp = pij21 + gij21 * temp - (ei1 - ej) * (ei1 - ej);  // pp2
                    temp = -Buffer::MeanField::rr2[mai1] * constants::PAULI_CPW - temp * constants::PAULI_CPH;
                    temp = smem_op2 ? constants::PAULI_EPSX : temp;
                    pf1 += temp > constants::PAULI_EPSX && ci1 == cj ? expf(temp) : 0.f;

                    // 2nd
                    temp = ci2 == cj ? 0.f : constants::MASS_DIF_2 / (ei2 + ej);
                    temp = temp * temp;
                    temp = pij22 + gij22 * temp - (ei2 - ej) * (ei2 - ej);  // pp2
                    temp = -Buffer::MeanField::rr2[mai2] * constants::PAULI_CPW - temp * constants::PAULI_CPH;
                    temp = smem_op1 ? constants::PAULI_EPSX : temp;
                    pf2 += temp > constants::PAULI_EPSX && ci1 == cj ? expf(temp) : 0.f;
                }
                __syncthreads();
            }
            pf1 = rsum32_(pf1);
            smem->boostx[warp_idx] = pf1;
            pf2 = rsum32_(pf2);
            smem->boosty[warp_idx] = pf2;
            __syncthreads();
            // 2nd step reduction
            pf1 = smem->boostx[lane_idx];
            pf1 = rsum32_(pf1);
            pf2 = smem->boosty[lane_idx];
            pf2 = rsum32_(pf2);
            // using leading thread
            if (!threadIdx.x) {
                temp = (pf1 - 1.f) * constants::PAULI_CPC;
                if (temp > curand_uniform(&rand_state[threadIdx.x + blockIdx.x * blockDim.x]))
                    smem->pauli_blocked = 1;
                temp = (pf2 - 1.f) * constants::PAULI_CPC;
                if (temp > curand_uniform(&rand_state[threadIdx.x + blockIdx.x * blockDim.x]))
                    smem->pauli_blocked = 1;
            }
            __syncthreads();
            int pauli_blocked = smem->pauli_blocked;
            __syncthreads();
            return (bool)pauli_blocked;
        }


    }
}