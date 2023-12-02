/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <iomanip>
#include <chrono>

#include "public/Test.h"

using namespace ckks;
using namespace std;

class Timer {
 public:
  Timer(const string& name) : name{name} {
    cudaDeviceSynchronize();
    CudaNvtxStart(name);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  void end() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    CudaNvtxStop();
    // cout << setprecision(3);
    // cout << name << ", " << fixed << setprecision(3) << milliseconds << " ms"
    //      << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  string name;
  cudaEvent_t start, stop;
  float milliseconds;
};

class Stopwatch
{
  public:
  Stopwatch(string timer_name) :
  name_(timer_name),
  start_time_(chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    auto end_time = chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::milliseconds>(end_time - start_time_);
    cout << "name_" << ": " << double(d.count()) << " ms" << endl;
  }
  private:
  string name_;
  chrono::high_resolution_clock::time_point start_time_;
};

class Benchmark {
 public:
  Benchmark(const Parameter& param) : ckks{param}, param{param} {
    ckks.context.EnableMemoryPool();
    
    // ModUpBench();
    // ModDownBench();
    // KeyswitchBench();
    Right_CCMult();
    // cout << "waht happen?" << endl;
    // Right_Rotate();
    // PtxtCtxtBatchBench();

    // the stage2 of the st-gcn in GPU accelerator

    // st_gcn_layer1_1();
    // st_gcn_layer1_2();
    // st_gcn_layer1_3();
    // st_gcn_layer2_1();
    // st_gcn_layer2_2();
    // st_gcn_layer2_3();
    // new_st_gcn_layer1_1();
  }


  template <typename F, typename R, class... Args>
  void Run(const string& message, R(F::*mf), Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      (ckks.context.*mf)(std::forward<Args>(args)...);
    }
  }

  template <typename Callable, class... Args>
  void Run(const string& message, Callable C, Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      C(std::forward<Args>(args)...);
    }
  }

  void RKSBench() {
    //Context context;
    
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    auto from = ckks.GetRandomPoly();
    DeviceVector ax, bx;
    DeviceVector cx = ckks.GetRandomPolyRNS(param.max_num_moduli_);
    DeviceVector to;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    Timer marker("A complete Keyswtich");
    // ckks.context.ModUp(from);
    // ckks.context.KeySwitch(from, key, ax, bx);
    // ckks.context.ModDown(cx, to, num_moduli_after_moddown);
    for (int i = 0; i < 100; ++i) {
      Run("FusedModUp", &Context::ModUp, from);
      Run("FusedKeySwitch", &Context::KeySwitch, from, key, ax, bx);
      Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
    }
  }

  void Right_CCMult() {
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;

    float T;
    for (int i = 0; i < 10; ++i) {
      Ciphertext op1 = ckks.GetRandomCiphertext();
      Ciphertext op2 = ckks.GetRandomCiphertext();
      Ciphertext out;
      Ciphertext ksto;
      DeviceVector d2;
      Ciphertext d2o;
      const int num_moduli_after_moddown = param.chain_length_;
      auto key = ckks.GetRandomKey();
      Timer marker("A complete CCMult");
      // ckks.context.CCMult(op1, op2, out, d2);
      Run("CCMult", &Context::CCMult, op1, op2, out, d2);
      auto d = ckks.context.ModUp(d2);
      Run("FusedKeySwitch", &Context::KeySwitch, d, key, d2o.ax__, d2o.bx__);
      Run("FusedModDown", &Context::ModDown, d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      Run("FusedModDown", &Context::ModDown, d2o.bx__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(out, ksto, out);
      marker.end();
      cout << marker.name << ", " << marker.milliseconds << " ms" << endl;
      T += marker.milliseconds;
    }
    cout << "total time: " << T << endl;
  }

  // (a, b) -> a', b'
  // KS(a', evk) -> a'', b''
  // (a'', b' + b'')
  void Right_Rotate() {
  ckks.context.is_modup_batched = true;
  ckks.context.is_moddown_fused = true;
  ckks.context.is_keyswitch_fused = true;
  float T;
  cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
  DeviceVector vec = ckks.GetRandomPolyRNS(1);
  for (int i = 0; i < 1; ++i) {
    Ciphertext ct = ckks.GetRandomCiphertext();
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    Timer marker("A complete Rotate");
    DeviceVector rax;
    DeviceVector rbx;
    Ciphertext tmp;
    Ciphertext IPo;
    Ciphertext Out;
    ckks.context.AutomorphismTransform(ct, rax, rbx, 1, vec);
    auto from = ckks.GetRandomPoly();
    cout << "from " << from.size() << endl; // 14 11
    cout << "rax " << rax.size() << endl;
    auto raxo = ckks.context.ModUp(rax);
    ckks.context.KeySwitch(raxo, key, IPo.ax__, IPo.bx__);
    ckks.context.ModDown(IPo.ax__, Out.ax__, num_moduli_after_moddown);
    ckks.context.ModDown(IPo.bx__, Out.bx__, num_moduli_after_moddown);
    tmp.bx__.append(rbx);
    ckks.context.Add(Out, tmp, Out);
    marker.end();
    cout << marker.name << ", " << marker.milliseconds << " ms" << endl;
    T += marker.milliseconds;
  }
  cout << "total time: " << T << endl;
} 

/*
  void st_gcn_layer1_1() {
    // excute the rescale ciphertext five times, now the L is 11
    // base setup
    float T;
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_;
    float T2;
    Timer mm("");
    auto key = ckks.GetRandomKey();
    Ciphertext ksto;
    DeviceVector d2;
    Ciphertext d2o;
    mm.end();
    T2 += mm.milliseconds;
    // data setup

    for (int i = 0; i < 16; ++i) {
      Timer mmm("");
      Ciphertext op1 = ckks.GetRandomCiphertext();
      Ciphertext out;
      mmm.end();
      T2 += mmm.milliseconds;
      
      Timer marker("1");
      for (int j = 0; j < 64; ++j) {
        ckks.context.CCMult(op1, op1, out, d2);
        ckks.context.ModUp(d2);
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        ckks.context.Add(out, ksto, out);
      }
      marker.end();
      T += marker.milliseconds;
    }

    cout << "st-gcn layer 1 cudamemcopy and init finished, consume " << T2 << " ms" << endl;
    cout << "st-gcn layer 1 conv1 rotation finished, consume " << T << " ms" << endl;

    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 16; ++i) {
        for (int r = 0; r < 2; ++r) {
          int batch_size = 64;
          vector<Ciphertext> op1(batch_size);
          vector<Plaintext> op2(batch_size);
          Plaintext bias1_1 = ckks.GetRandomPlaintext();
          Ciphertext accum;
          // setup
          for (int i = 0; i < batch_size; i++) {
            op1[i] = ckks.GetRandomCiphertext();
            op2[i] = ckks.GetRandomPlaintext();
          }

          Timer m("2");
          MultPtxtBatch batcher(&ckks.context);
          
          for (int i = 0; i < batch_size; i++) {
            batcher.push(op1[i], op2[i]);
          }
          batcher.flush(accum);
          // todo rescale (10)
          ckks.context.CCAdd(accum, bias1_1, accum); // c+p
          m.end();
          T += m.milliseconds;
        }
      }
    }

    cout << "st-gcn layer 1 conv1 finished, consume " << T << " ms" << endl;
  }


  void new_st_gcn_layer1_1() {
    // excute the rescale ciphertext five times, now the L is 11
    // base setup

    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    float T;
    Timer s("");
    // data setup
    vector<Ciphertext> st_gcn0_output(16);
    vector<vector<vector<vector<Ciphertext>>>> cts_conv1_1(3, vector<vector<vector<Ciphertext>>>(
                                                           16, vector<vector<Ciphertext>>(
                                                           2, vector<Ciphertext>(64))));
    vector<vector<vector<Plaintext>>> conv1_1_plain(3, vector<vector<Plaintext>>(
                                                           2, vector<Plaintext>(64)));
    vector<vector<vector<Ciphertext>>> cts_temp_1(3, vector<vector<Ciphertext>>(
                                                           16, vector<Ciphertext>(2)));
    vector<vector<Plaintext>> bias1_1_plain(3, vector<Plaintext>(2));                                                       
    Ciphertext temp;
    DeviceVector d2 = ckks.GetRandomPoly();
    Ciphertext d2o;
    Ciphertext ksto;
    s.end();

    // cout << "break_point1" << endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        bias1_1_plain[i][j] = ckks.GetRandomPlaintext();
      }
    }
    for (int k = 0; k < 3; ++k) {
      for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 64; ++j) {
          conv1_1_plain[k][r][j] = ckks.GetRandomPlaintext();
        }
      }
    }
    T += s.milliseconds;
    for (int i = 0; i < 16; i++) {
      st_gcn0_output[i] = ckks.GetRandomCiphertext();
    }
    // cout << "break_point2" << endl;
    //Timer marker("newe st-gcn layer 1_1");
    Timer s2("");
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 64; ++j) {
        // d2.append(st_gcn0_output[i].ax__);
        // cout << 1;
        ckks.context.ModUp(d2);
        // cout << 2;
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        // cout << 3;
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        // cout << 4;
        ckks.context.Add(st_gcn0_output[i], ksto, temp);
        // cout << j << endl;
        for (int k = 0; k < 3; ++k) {
          for (int r= 0; r < 2; ++r) {
            cts_conv1_1[k][i][r][j].ax__.append(temp.ax__);
            cts_conv1_1[k][i][r][j].bx__.append(temp.bx__);
          }
        }
      }
    }
    // cout << "break_point3" << endl;
    cout << "st-gcn layer 1 conv1 rotation finished" << endl;

    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 16; ++i) {
        for (int r = 0; r < 2; ++r) {
          // cout << 1 << endl;
          MultPtxtBatch batcher(&ckks.context);
          for (int j = 0; j < 64; j++) {
            batcher.push(cts_conv1_1[k][i][r][j], conv1_1_plain[k][r][j]);
          }
          // cout << 2 << endl;
          batcher.flush(cts_temp_1[k][i][r]);
          // cout << 3 << endl;
          ckks.context.CCAdd(cts_temp_1[k][i][r], bias1_1_plain[k][r], cts_temp_1[k][i][r]);
        }
      }
    }
    s2.end();
    T += s2.milliseconds;
    cout << "st-gcn layer 1 conv1 finished, consume " << T << endl;
  }

 
  void st_gcn_layer1_2() {
    // excute the rescale ciphertext five times, now the L is 10
    vector<int>xindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>yindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>xindex1_0 {12,0,3,16,5,6,7,16,9,10,11,16,13,14,15,16};
    vector<int>yindex1_0 {0,1,2,16,4,5,6,16,8,9,10,16,12,13,14,16};
    vector<int>xindex1_1 {16,16,16,16,16,16,5,6,16,16,9,10,0,12,13,14};
    vector<int>yindex1_1 {16,16,16,16,16,16,6,7,16,16,10,11,12,13,14,15};
    vector<int>xindex2   {1,16,16,2,16,4,16,16,16,8,16,16,16,16,16,16};
    vector<int>yindex2   {0,16,16,3,16,5,16,16,16,9,16,16,16,16,16,16};
    vector<vector<int>> index_set{xindex0, xindex1_0, xindex1_1, xindex2};
    float T;
    for (int i = 0; i < 16; ++i) {
      for (int r = 0; r < 2; ++r) {
        int cnt = 0;
        for (int v = 0; v < 4; ++v) {
          if (index_set[v][i] < 16) {
            cnt++;
          }
        }
        int batch_size = cnt;
        vector<Ciphertext> op1(batch_size);
        vector<Plaintext> op2(batch_size);
        Ciphertext accum;
        Plaintext b1_1 = ckks.GetRandomPlaintext();
          // setup
        for (int i = 0; i < batch_size; i++) {
          op1[i] = ckks.GetRandomCiphertext();
          op2[i] = ckks.GetRandomPlaintext();
        }

        Timer m("1");
        MultPtxtBatch batcher(&ckks.context);
        for (int i = 0; i < batch_size; i++) {
          batcher.push(op1[i], op2[i]);
        }
        batcher.flush(accum);
        // todo rescale (9)
        ckks.context.CCAdd(accum, b1_1, accum);
        m.end();
        T += m.milliseconds;
      }
    }

    cout << "st-gcn layer 1 A finished, consume " << T << " ms" << endl;
  }

  void new_st_gcn_layer1_2() {
    // excute the rescale ciphertext five times, now the L is 10
    vector<vector<Ciphertext>> cts_A_1(16, vector<Ciphertext>(2));
    vector<vector<vector<Ciphertext>>> cts_preA_1 (4, vector<vector<Ciphertext>>( 
                                                  2, vector<Ciphertext>(16)));
    vector<vector<vector<Plaintext>>> A_plain_0 (4, vector<vector<Plaintext>>( 
                                                  16, vector<Plaintext>(2)));
    vector<Plaintext> b1_1_plain(2);
    for (int i = 0; i < 2; ++i) {
      b1_1_plain[i] = ckks.GetRandomPlaintext();
    }
    for (int k = 0; k < 4; ++k) {
      for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < 16; ++i) {
          cts_preA_1[k][r][i] = ckks.GetRandomCiphertext();
          A_plain_0[k][i][r] = ckks.GetRandomPlaintext();
        }
      }
    }
    vector<int>xindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>yindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>xindex1_0 {12,0,3,16,5,6,7,16,9,10,11,16,13,14,15,16};
    vector<int>yindex1_0 {0,1,2,16,4,5,6,16,8,9,10,16,12,13,14,16};
    vector<int>xindex1_1 {16,16,16,16,16,16,5,6,16,16,9,10,0,12,13,14};
    vector<int>yindex1_1 {16,16,16,16,16,16,6,7,16,16,10,11,12,13,14,15};
    vector<int>xindex2   {1,16,16,2,16,4,16,16,16,8,16,16,16,16,16,16};
    vector<int>yindex2   {0,16,16,3,16,5,16,16,16,9,16,16,16,16,16,16};
    vector<vector<int>> index_set{xindex0, xindex1_0, xindex1_1, xindex2};
    for (int i = 0; i < 16; ++i) {
      for (int r = 0; r < 2; ++r) {
        MultPtxtBatch batcher(&ckks.context);
        for (int v = 0; v < 4; ++v) {
          if (index_set[v][i] < 16) {
            batcher.push(cts_preA_1[v][index_set[v][i]][r], A_plain_0[v][i][r]);
          }
        }
        batcher.flush(cts_A_1[i][r]);
        // todo rescale (9)
        ckks.context.CCAdd(cts_A_1[i][r], b1_1_plain[r], cts_A_1[i][r]);
      }
    }

    cout << "st-gcn layer 1 A finished" << endl;
  }

 
  void st_gcn_layer1_3() {
    // excute the rescale ciphertext five times, now the L is 9
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    float T;
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 2; ++j) {
        Ciphertext op1 = ckks.GetRandomCiphertext();
        Ciphertext op2 = ckks.GetRandomCiphertext();
        Ciphertext out;
        Ciphertext ksto;
        DeviceVector d2;
        auto from = ckks.GetRandomPoly();
        Ciphertext d2o;
        const int num_moduli_after_moddown = param.chain_length_;
        auto key = ckks.GetRandomKey();
        Timer m("");
        for (int k = 0; k < 9; ++k) {
          ckks.context.CCMult(op1, op2, out, d2);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(out, ksto, out);
        }
        m.end();
        T += m.milliseconds;
      }
    }  

    cout << "baby-step ciphertext rotation end, consume " << T << " ms" << endl;

    for (int i = 0; i < 16; ++i) {
      for (int m = 0; m < 2; ++m) {
        for (int r = 0; r < 2; ++r) {
          for (int k = 0; k < 64; ++k) {
            // kswtich setup
            Ciphertext op1 = ckks.GetRandomCiphertext();
            Ciphertext op2 = ckks.GetRandomCiphertext();
            Ciphertext out;
            Ciphertext ksto;
            DeviceVector d2;
            auto from = ckks.GetRandomPoly();
            Ciphertext d2o;
            const int num_moduli_after_moddown = param.chain_length_;
            auto key = ckks.GetRandomKey();

            // MAC setup
            int batch_size = 9;
            vector<Ciphertext> oop1(batch_size);
            vector<Plaintext> oop2(batch_size);
            // setup
            for (int i = 0; i < batch_size; i++) {
              oop1[i] = ckks.GetRandomCiphertext();
              oop2[i] = ckks.GetRandomPlaintext();
            }
            Timer m2("");
            MultPtxtBatch batcher(&ckks.context);
            Ciphertext accum;
            for (int i = 0; i < batch_size; i++) {
              batcher.push(oop1[i], oop2[i]);
            }
            batcher.flush(accum);

            ckks.context.CCMult(op1, op2, out, d2);
            ckks.context.ModUp(d2);
            ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
            ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
            ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
            ckks.context.Add(out, ksto, out);
            m2.end();
            T += m2.milliseconds;
          }
          // setup
          // cout << 111111111 << endl;
          int batch_size = 64;
          vector<Ciphertext> op(batch_size);
          Ciphertext accum;
          for (int i = 0; i < 64; i++) {
            op[i] = ckks.GetRandomCiphertext();
          }
          // addmany * 64
          Timer m3("");
          ckks.context.Add(op[0], op[1], accum);
          for (int j = 2; j < 64; ++j) {
              ckks.context.Add(accum, op[i], accum);
          }
          m3.end();
          T += m3.milliseconds;
        }
        // setup
        int batch_size = 2;
        vector<Ciphertext> op(batch_size);
        Plaintext bias2 = ckks.GetRandomPlaintext();
        Ciphertext accum;
        for (int i = 0; i < batch_size; i++) {
          op[i] = ckks.GetRandomCiphertext();
        }

        Timer m4("");
        // addmany * 2
        ckks.context.Add(op[0], op[1], accum);

        // todo rescale (8)
        ckks.context.CCAdd(accum, bias2, accum);
        m4.end();
        T += m4.milliseconds;
      }
    }

    cout << "st-gcn layer 1, conv2, bn2 finished, consume " << T << " ms" << endl;
  }

  void new_st_gcn_layer1_3() {
    // excute the rescale ciphertext five times, now the L is 9
    // base setup
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    // data setup
    vector<vector<vector<vector<vector<Ciphertext>>>>> conv2_1_temp(16, vector<vector<vector<vector<Ciphertext>>>>(
                                                                    2, vector<vector<vector<Ciphertext>>>(
                                                                    2, vector<vector<Ciphertext>>(
                                                                    64, vector<Ciphertext>(9)))));
    vector<vector<vector<vector<Ciphertext>>>> conv2_1_rotated_extra(16, vector<vector<vector<Ciphertext>>>(
                                                           2, vector<vector<Ciphertext>>(
                                                           2, vector<Ciphertext>(64))));
    vector<vector<vector<vector<Plaintext>>>> conv2_1_plain(2, vector<vector<vector<Plaintext>>>(
                                                           2, vector<vector<Plaintext>>(
                                                           64, vector<Plaintext>(9))));
    vector<vector<vector<Ciphertext>>> conv2_1_rotated_extra_temp(16, vector<vector<Ciphertext>>(
                                                           2, vector<Ciphertext>(2)));
    vector<vector<vector<Ciphertext>>> conv2_1_rotated_intra(16, vector<vector<Ciphertext>>(
                                                           2, vector<Ciphertext>(9)));
    vector<vector<Ciphertext>> cts_conv2_1(16, vector<Ciphertext>(2)); 
    vector<vector<Ciphertext>> cts_A_1(16, vector<Ciphertext>(2));
    vector<Plaintext> bias2_1_plain(2);
    for (int i = 0; i < 2; ++i) {
      bias2_1_plain[i] = ckks.GetRandomPlaintext();
    }
    for (int m = 0; m < 2; ++m) {
      for (int r = 0; r < 2; ++r) {
        for (int k = 0; k < 64; +k) {
          for (int j = 0; j < 9; ++j) {
            conv2_1_plain[m][r][k][j] = ckks.GetRandomPlaintext();
          }
        }
      }
    }
    for (int k = 0; k < 16; ++k) {
      for (int r = 0; r < 2; ++r) {
          cts_A_1[k][r] = ckks.GetRandomCiphertext();
      }
    }

    DeviceVector d2;
    Ciphertext d2o;
    Ciphertext ksto; 


    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int r = 0; r < 9; ++r) {
          d2.append(cts_A_1[i][j].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(cts_A_1[i][j], ksto, conv2_1_rotated_intra[i][j][r]);
        }
      }
    }  

    cout << "baby-step ciphertext rotation end" << endl;

    for (int i = 0; i < 16; ++i) {
      for (int m = 0; m < 2; ++m) {
        for (int r = 0; r < 2; ++r) {
          for (int k = 0; k < 64; ++k) {
            MultPtxtBatch batcher(&ckks.context);
            for (int j = 0; j < 9; ++j) {
              conv2_1_temp[i][m][r][k][j].ax__.append(conv2_1_rotated_intra[i][r][j].ax__);
              conv2_1_temp[i][m][r][k][j].bx__.append(conv2_1_rotated_intra[i][r][j].bx__);
              batcher.push(conv2_1_temp[i][m][r][k][j], conv2_1_plain[m][r][k][j]);
            }
            batcher.flush(conv2_1_rotated_extra[i][m][r][k]);
            d2.append(conv2_1_rotated_extra[i][m][r][k].ax__);
            ckks.context.ModUp(d2);
            ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
            ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
            ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
            //ckks.context.Add(conv2_1_rotated_extra[i][m][r][k], ksto, conv2_1_rotated_extra[i][j][r][k]);
          }
          for (int n = 0; n < 64; n++) {
            ckks.context.Add(conv2_1_rotated_extra_temp[i][m][r], conv2_1_rotated_extra[i][m][r][n], conv2_1_rotated_extra_temp[i][m][r]);
          }
        }
        ckks.context.Add(conv2_1_rotated_extra_temp[i][m][0], conv2_1_rotated_extra_temp[i][m][1], cts_conv2_1[i][m]);
        // todo rescale (8)
        ckks.context.CCAdd(cts_conv2_1[i][m], bias2_1_plain[m], cts_conv2_1[i][m]);
      }
    }

    cout << "st-gcn layer 1, conv2, bn2 finished" << endl;
  }

  
  void st_gcn_layer2_1() {
    // excute the rescale ciphertext five times, now the L is 8
    float T;
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    for (int i = 0; i < 16; ++i) {
      Ciphertext op1 = ckks.GetRandomCiphertext();
      Ciphertext op2 = ckks.GetRandomCiphertext();
      Ciphertext out;
      Ciphertext ksto;
      DeviceVector d2;
      auto from = ckks.GetRandomPoly();
      Ciphertext d2o;
      const int num_moduli_after_moddown = param.chain_length_;
      auto key = ckks.GetRandomKey();
      Timer m("");
      for (int j = 0; j < 64; ++j) {
        ckks.context.CCMult(op1, op2, out, d2);
        ckks.context.ModUp(d2);
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        ckks.context.Add(out, ksto, out);
      }
      m.end();
      T += m.milliseconds;
    }


    cout << "st-gcn layer 2 conv1 rotation finished, consume " << T << " ms" << endl;
    // ks setup
    Ciphertext op1 = ckks.GetRandomCiphertext();
    Ciphertext op2 = ckks.GetRandomCiphertext();
    Ciphertext out;
    Ciphertext ksto;
    DeviceVector d2;
    auto from = ckks.GetRandomPoly();
    Ciphertext d2o;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();

    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 16; ++i) {
        for (int r = 0; r < 2; ++r) {
          // rotate setup
          Ciphertext op1 = ckks.GetRandomCiphertext();
          Ciphertext op2 = ckks.GetRandomCiphertext();
          Ciphertext out;
          Ciphertext ksto;
          DeviceVector d2;
          auto from = ckks.GetRandomPoly();
          Ciphertext d2o;
          const int num_moduli_after_moddown = param.chain_length_;
          auto key = ckks.GetRandomKey();
          // addmany setup
          int batch_size = 64;
          vector<Ciphertext> oop1(batch_size);
          vector<Plaintext> oop2(batch_size);
          Plaintext bias = ckks.GetRandomPlaintext();
          Ciphertext a1 = ckks.GetRandomCiphertext();
          Ciphertext a2 = ckks.GetRandomCiphertext();
          // add setup

          for (int i = 0; i < batch_size; i++) {
            oop1[i] = ckks.GetRandomCiphertext();
            oop2[i] = ckks.GetRandomPlaintext();
          }

          Timer m2("");
          MultPtxtBatch batcher(&ckks.context);
          Ciphertext accum;
          for (int i = 0; i < batch_size; i++) {
            batcher.push(oop1[i], oop2[i]);
          }
          batcher.flush(accum);

          // todo rescale (7)

          ckks.context.CCMult(op1, op2, out, d2);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(out, ksto, out);

          ckks.context.Add(a1, a2, out);

          ckks.context.CCMult(op1, op2, out, d2);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(out, ksto, out);
          m2.end();
          T += m2.milliseconds;
        }
         // addmany setup
        int batch_size = 2;
        vector<Ciphertext> op(batch_size);
        Plaintext bias2 = ckks.GetRandomPlaintext();
        Ciphertext accum;
        for (int i = 0; i < batch_size; i++) {
          op[i] = ckks.GetRandomCiphertext();
        }
        // addmany * 2

        Timer m3("");
        ckks.context.Add(op[0], op[1], accum);

        // todo rescale (6)
        ckks.context.CCAdd(accum, bias2, accum);
        m3.end();
        T += m3.milliseconds;
      }
    }

    cout << "st-gcn layer 2 conv1 finished, consume " << T << " ms" << endl;
  }

  void new_st_gcn_layer2_1() {
    // excute the rescale ciphertext five times, now the L is 8
    // base setup
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    // data setup
    vector<Ciphertext> st_gcn1_output(16);
    vector<vector<vector<vector<Ciphertext>>>> cts_conv1_2(3, vector<vector<vector<Ciphertext>>>(
                                                           16, vector<vector<Ciphertext>>(
                                                           2, vector<Ciphertext>(64))));
    vector<vector<vector<Plaintext>>> conv1_2_plain(3, vector<vector<Plaintext>>(
                                                           2, vector<Plaintext>(64)));
    vector<vector<vector<Ciphertext>>> cts_temp_2(3, vector<vector<Ciphertext>>(
                                                           16, vector<Ciphertext>(2)));
    vector<Plaintext> bias1_2_plain(3);       
    vector<vector<Ciphertext>> cts_conv1_temp(3, vector<Ciphertext>(16));                                                
    Ciphertext temp;
    DeviceVector d2;
    Ciphertext d2o;
    Ciphertext ksto;
    Plaintext mask = ckks.GetRandomPlaintext();

    for (int i = 0; i < 16; i++) {
      st_gcn1_output[i] = ckks.GetRandomCiphertext();
    }
    for (int k = 0; k < 3; ++k) {
      for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 64; ++j) {
          conv1_2_plain[k][r][j] = ckks.GetRandomPlaintext();
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      bias1_2_plain[i] = ckks.GetRandomPlaintext();
    }

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 64; ++j) {
        d2.append(st_gcn1_output[i].ax__);
        ckks.context.ModUp(d2);
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        ckks.context.Add(st_gcn1_output[i], ksto, temp);
        for (int k = 0; k < 3; ++k) {
          for (int r= 0; r < 2; ++r) {
            cts_conv1_2[k][i][r][j].ax__.append(temp.ax__);
            cts_conv1_2[k][i][r][j].bx__.append(temp.bx__);
          }
        }
      }
    }

    cout << "st-gcn layer 2 conv1 rotation finished" << endl;

    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 16; ++i) {
        for (int r = 0; r < 2; ++r) {
          MultPtxtBatch batcher(&ckks.context);
          for (int j = 0; j < 64; j++) {
            batcher.push(cts_conv1_2[k][i][r][j], conv1_2_plain[k][r][j]);
          }
          batcher.flush(cts_temp_2[k][i][r]);
          // todo rescale (7)
          d2.append(cts_temp_2[k][i][r].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(cts_temp_2[k][i][r], ksto, temp);

          ckks.context.Add(cts_temp_2[k][i][r], temp, cts_temp_2[k][i][r]);
          ckks.context.PMult(cts_temp_2[k][i][r], mask, cts_temp_2[k][i][r]);

          d2.append(cts_temp_2[k][i][r].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(cts_temp_2[k][i][r], ksto, cts_temp_2[k][i][r]);
        }
        ckks.context.Add(cts_temp_2[k][i][0], cts_temp_2[k][i][1], cts_conv1_temp[k][i]);
        // todo rescale (6)
        ckks.context.CCAdd(cts_conv1_temp[k][i], bias1_2_plain[k], cts_conv1_temp[k][i]);
      }
    }

    cout << "st-gcn layer 2 conv1 finished" << endl;
  }

 
  void st_gcn_layer2_2() {
    // excute the rescale ciphertext five times, now the L is 6
    float T;
    vector<int>xindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>yindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>xindex1_0 {12,0,3,16,5,6,7,16,9,10,11,16,13,14,15,16};
    vector<int>yindex1_0 {0,1,2,16,4,5,6,16,8,9,10,16,12,13,14,16};
    vector<int>xindex1_1 {16,16,16,16,16,16,5,6,16,16,9,10,0,12,13,14};
    vector<int>yindex1_1 {16,16,16,16,16,16,6,7,16,16,10,11,12,13,14,15};
    vector<int>xindex2   {1,16,16,2,16,4,16,16,16,8,16,16,16,16,16,16};
    vector<int>yindex2   {0,16,16,3,16,5,16,16,16,9,16,16,16,16,16,16};
    vector<vector<int>> index_set{xindex0, xindex1_0, xindex1_1, xindex2};
    for (int i = 0; i < 16; ++i) {
      int cnt = 0;
      for (int v = 0; v < 4; ++v) {
        if (index_set[v][i] < 16) {
          cnt++;
        }
      }
      int batch_size = cnt;
      vector<Ciphertext> op1(batch_size);
      vector<Plaintext> op2(batch_size);
      Plaintext b1 = ckks.GetRandomPlaintext();
      Ciphertext out;
        // setup
      for (int i = 0; i < batch_size; i++) {
        op1[i] = ckks.GetRandomCiphertext();
        op2[i] = ckks.GetRandomPlaintext();
      }
      Timer m("");
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
      // rescale (now L = 5)

      ckks.context.CCAdd(op1[0], b1, out);
      m.end();
      T += m.milliseconds;
    }

    cout << "st-gcn layer 2 A finished, consume " << T << " ms" << endl;
  }

  void new_st_gcn_layer2_2() {
    // excute the rescale ciphertext five times, now the L is 6
    vector<Ciphertext> cts_A_2(16);
    vector<vector<Ciphertext>> cts_preA_2 (4,vector<Ciphertext>(16));
    vector<vector<Plaintext>> A_plain_2 (4, vector<Plaintext>(16));
    Plaintext b1_2_plain = ckks.GetRandomPlaintext();
    for (int k = 0; k < 4; ++k) {
      for (int i = 0; i < 16; ++i) {
        cts_preA_2[k][i] = ckks.GetRandomCiphertext();
        A_plain_2[k][i] = ckks.GetRandomPlaintext();
      }
    }
    vector<int>xindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>yindex0   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<int>xindex1_0 {12,0,3,16,5,6,7,16,9,10,11,16,13,14,15,16};
    vector<int>yindex1_0 {0,1,2,16,4,5,6,16,8,9,10,16,12,13,14,16};
    vector<int>xindex1_1 {16,16,16,16,16,16,5,6,16,16,9,10,0,12,13,14};
    vector<int>yindex1_1 {16,16,16,16,16,16,6,7,16,16,10,11,12,13,14,15};
    vector<int>xindex2   {1,16,16,2,16,4,16,16,16,8,16,16,16,16,16,16};
    vector<int>yindex2   {0,16,16,3,16,5,16,16,16,9,16,16,16,16,16,16};
    vector<vector<int>> index_set{xindex0, xindex1_0, xindex1_1, xindex2};
    for (int i = 0; i < 16; ++i) {
      MultPtxtBatch batcher(&ckks.context);
      for (int v = 0; v < 4; ++v) {
        if (index_set[v][i] < 16) {
          batcher.push(cts_preA_2[v][index_set[v][i]], A_plain_2[v][i]);
        }
      }
      batcher.flush(cts_A_2[i]);
      // todo rescale (5)
      ckks.context.CCAdd(cts_A_2[i], b1_2_plain, cts_A_2[i]);
    }
    cout << "st-gcn layer 2 A finished" << endl;
  }

  void st_gcn_layer2_3() {
    // excute the rescale ciphertext five times, now the L is 5
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    float T;
    for (int i = 0; i < 16; ++i) {
      Ciphertext op1 = ckks.GetRandomCiphertext();
      Ciphertext op2 = ckks.GetRandomCiphertext();
      Ciphertext out;
      Ciphertext ksto;
      DeviceVector d2;
      auto from = ckks.GetRandomPoly();
      Ciphertext d2o;
      const int num_moduli_after_moddown = param.chain_length_;
      auto key = ckks.GetRandomKey();
      Timer m("");
      for (int j = 0; j < 9; ++j) {
        ckks.context.CCMult(op1, op2, out, d2);
        ckks.context.ModUp(d2);
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        ckks.context.Add(out, ksto, out);
      }
      m.end();
      T += m.milliseconds;
    }  

    cout << "baby-step ciphertext rotation end, consume " << T << " ms" << endl;
    // ks setup
    Ciphertext op1 = ckks.GetRandomCiphertext();
    Ciphertext op2 = ckks.GetRandomCiphertext();
    Ciphertext out;
    Ciphertext ksto;
    DeviceVector d2;
    auto from = ckks.GetRandomPoly();
    Ciphertext d2o;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();

    for (int i = 0; i < 16; ++i) {
      auto key = ckks.GetRandomKey();
      for (int r = 0; r < 2; ++r) {
        Ciphertext op1 = ckks.GetRandomCiphertext();
        Ciphertext op2 = ckks.GetRandomCiphertext();
        Ciphertext out;
        Ciphertext ksto;
        DeviceVector d2;
        auto from = ckks.GetRandomPoly();
        Ciphertext d2o;
        const int num_moduli_after_moddown = param.chain_length_;
        auto key = ckks.GetRandomKey();
        for (int k = 0; k < 64; ++k) {
          // * after + setup
          int batch_size = 9;
          vector<Ciphertext> oop1(batch_size);
          vector<Plaintext> oop2(batch_size);
          // setup
          for (int i = 0; i < batch_size; i++) {
            oop1[i] = ckks.GetRandomCiphertext();
            oop2[i] = ckks.GetRandomPlaintext();
          }
          Timer m1("");
          MultPtxtBatch batcher(&ckks.context);
          Ciphertext accum;
          for (int i = 0; i < batch_size; i++) {
            batcher.push(oop1[i], oop2[i]);
          }
          batcher.flush(accum);

          //rotate_vector
          ckks.context.CCMult(op1, op2, out, d2);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(out, ksto, out);
          m1.end();
          T += m1.milliseconds;
        }
        // setup
        int batch_size = 64;
        vector<Ciphertext> op(batch_size);
        Ciphertext accum;
        for (int i = 0; i < 64; i++) {
          op[i] = ckks.GetRandomCiphertext();
        }
        // addmany * 64
        Timer m2("");
        ckks.context.Add(op[0], op[1], accum);
        for (int j = 2; j < 64; ++j) {
            ckks.context.Add(accum, op[i], accum);
        }
        m2.end();
        T += m2.milliseconds;
        // todo rescale (4)
      }
      // rotate_vector
      Timer m3("");
      ckks.context.CCMult(op1, op2, out, d2);
      ckks.context.ModUp(d2);
      ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
      ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(out, ksto, out);
      m3.end();
      T += m3.milliseconds;

      // setup
      int batch_size = 2;
      vector<Ciphertext> op(batch_size);
      Plaintext bias2 = ckks.GetRandomPlaintext();
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        op[i] = ckks.GetRandomCiphertext();
      }

      Timer m4("");
      // addmany * 2
      ckks.context.Add(op[0], op[1], accum);
      // _plain
      ckks.context.PMult(accum, bias2, accum);
      // rotate_vector
      ckks.context.CCMult(op1, op2, out, d2);
      ckks.context.ModUp(d2);
      ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
      ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(out, ksto, out);
      m4.end();
      T += m4.milliseconds;
    }
    // setup
    int batch_size = 2;
    vector<Ciphertext> op(batch_size);
    Plaintext bias2_2 = ckks.GetRandomPlaintext();
    Ciphertext accum;
    for (int i = 0; i < batch_size; i++) {
      op[i] = ckks.GetRandomCiphertext();
    }
    // addmany * 2
    Timer m5("");
    ckks.context.Add(op[0], op[1], accum);
    
    // todo rescale (3)

    ckks.context.CCAdd(accum, bias2_2, accum);
    m5.end();
    T += m5.milliseconds;
    cout << "st-gcn layer 2, conv2, bn2 finished, consume " << T << " ms" << endl;
  }

  void new_st_gcn_layer2_3() {
    // current  L = 5
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    vector<Ciphertext> cts_A_2(16);
    vector<vector<Ciphertext>> cts_conv2_2(16, vector<Ciphertext>(2));
    vector<vector<Ciphertext>> conv2_2_rotated_intra(16, vector<Ciphertext>(9));
    vector<vector<vector<vector<Ciphertext>>>> conv2_2_temp(16, vector<vector<vector<Ciphertext>>> (
                                                            2, vector<vector<Ciphertext>> (
                                                            64, vector<Ciphertext>(9))));
    vector<vector<vector<Ciphertext>>> conv2_2_rotated_extra(16, vector<vector<Ciphertext>> (
                                                            2, vector<Ciphertext>(64)));
    vector<Ciphertext> cts_conv2_2_final(16);
    vector<vector<vector<Plaintext>>> conv2_2_plain(2, vector<vector<Plaintext>>(64, vector<Plaintext>(9)));

    for (int i = 0; i < 16; ++i) {
      cts_A_2[i] = ckks.GetRandomCiphertext();
    }
    for (int r = 0; r < 2; ++r) {
      for (int k = 0; k < 64; ++k) {
        for (int j = 0; j < 9; ++j) {
          conv2_2_plain[r][k][j] = ckks.GetRandomPlaintext();
        }
      }
    }
    Plaintext mask_plain = ckks.GetRandomPlaintext();
    Plaintext bias2_2_plain = ckks.GetRandomPlaintext();
    DeviceVector d2;
    Ciphertext d2o;
    Ciphertext ksto;
    Ciphertext rotated_temp;

    for (int i = 0; i < 16; ++i) {
      for (int r = 0; r < 9; ++r) {
        d2.append(cts_A_2[i].ax__);
        ckks.context.ModUp(d2);
        ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
        ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
        ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
        ckks.context.Add(cts_A_2[i], ksto, conv2_2_rotated_intra[i][r]);
      }
    }  

    cout << "baby-step ciphertext rotation end" << endl;

    for (int i = 0; i < 16; ++i) {
      for (int r = 0; r < 2; ++r) {
        for (int k = 0; k < 64; ++k) {
          MultPtxtBatch batcher(&ckks.context);
          for (int j = 0; j < 9; ++j) {
            conv2_2_temp[i][r][k][j].ax__.append(conv2_2_rotated_intra[i][j].ax__);
            conv2_2_temp[i][r][k][j].bx__.append(conv2_2_rotated_intra[i][j].bx__);
            batcher.push(conv2_2_temp[i][r][k][j], conv2_2_plain[r][k][j]);
          }
          batcher.flush(conv2_2_rotated_extra[i][r][k]);
          d2.append(conv2_2_rotated_extra[i][r][k].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(conv2_2_rotated_extra[i][r][k], ksto, conv2_2_rotated_extra[i][r][k]);
        }
        for (int m = 0; m < 64; ++m) {
          ckks.context.Add(cts_conv2_2[i][r], conv2_2_rotated_extra[i][r][m], cts_conv2_2[i][r]);
        }
        // todo rescale (4)

          d2.append(cts_conv2_2[i][r].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(cts_conv2_2[i][r], ksto, rotated_temp);

          ckks.context.Add(cts_conv2_2[i][r], rotated_temp, cts_conv2_2[i][r]);
          ckks.context.PMult(cts_conv2_2[i][r], mask_plain, cts_conv2_2[i][r]);

          d2.append(cts_conv2_2[i][r].ax__);
          ckks.context.ModUp(d2);
          ckks.context.KeySwitch(d2, key, d2o.ax__, d2o.bx__);
          ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
          ckks.context.ModDown(d2o.ax__, ksto.bx__, num_moduli_after_moddown);
          ckks.context.Add(cts_conv2_2[i][r], ksto, cts_conv2_2[i][r]);
      }
      ckks.context.Add(cts_conv2_2[i][0], cts_conv2_2[i][1], cts_conv2_2_final[i]);
      // todo rescale (3)
      ckks.context.CCAdd(cts_conv2_2_final[i], bias2_2_plain, cts_conv2_2_final[i]);
      cout << "st-gcn layer 2, conv2, bn2 finished" << endl;
    }
  }

  void st_gcn_fc() {
    // current  L = 3
    // kswtich setup
    ckks.context.is_modup_batched = true;
    ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    auto from = ckks.GetRandomPoly();
    DeviceVector ax, bx;
    DeviceVector cx = ckks.GetRandomPolyRNS(param.max_num_moduli_);
    DeviceVector to;
    const int num_moduli_after_moddown = param.chain_length_;
    auto key = ckks.GetRandomKey();
    // addmany*16 setup
    int batch_size = 2;
    vector<Ciphertext> op(batch_size);
    // Plaintext fc_bias = ckks.GetRandomPlaintext();
    Plaintext fc_weight = ckks.GetRandomPlaintext();
    Ciphertext accum;
    for (int i = 0; i < batch_size; i++) {
      op[i] = ckks.GetRandomCiphertext();
    }
    // addmany * 16
    for (int i = 0; i < batch_size; i++) {
      ckks.context.Add(accum, op[i], accum);
    }

    // rorate_vector(accum)
    {
      // galosi
      Run("FusedModUp", &Context::ModUp, from);
      Run("FusedKeySwitch", &Context::KeySwitch, from, key, ax, bx);
      Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
      // add
    }
    ckks.context.Add(accum, op[1], accum);

    for (int i = 0; i < 4; ++i) {
      // rorate_vector(accum)
      {
        // galosi
        Run("FusedModUp", &Context::ModUp, from);
        Run("FusedKeySwitch", &Context::KeySwitch, from, key, ax, bx);
        Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
        // add
      }
      ckks.context.Add(accum, op[i], accum);
    }
    ckks.context.PMult(accum, fc_weight, accum);

    for (int i = 0; i < 6; ++i) {
      // rorate_vector(accum)
      {
        // galosi
        Run("FusedModUp", &Context::ModUp, from);
        Run("FusedKeySwitch", &Context::KeySwitch, from, key, ax, bx);
        Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
        // add
      }
      ckks.context.Add(accum, op[i], accum);
    }
  }
*/


// write a application like that
// ch1, ch2,
// kernel 1, kernel 2, kernel 3, kernel 4
  void HE_Conv_baseline() {
    // ckks.context.is_modup_batched = true;
    // ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_; // init l
    auto key = ckks.GetRandomKey();
    Ciphertext op1 = ckks.GetRandomCiphertext(); // init ciphertext(ch1, ch2)
    vector<vector<Plaintext>> m(4, vector<Plaintext>(9));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 9; ++j) {
        m[i][j] = ckks.GetRandomPlaintext();
      }
    }

    Ciphertext in_raise;
    Ciphertext d2o;
    Ciphertext ksto;
    vector<Ciphertext> out(9);
    Ciphertext temp;
    vector<Ciphertext> res(2);

    
    // 1-stage rot * 8 (autom -> modup -> innerp -> moddown --> add)
    out.push_back(op1); // record itself
    for (int i = 0; i < 8; ++i) {
      // automorphism replace it with ccmult
      ckks.context.CCMult(op1, op1, temp, in_raise);
      DeviceVector after_modup = ckks.context.ModUp(in_raise);
      ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
      ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(temp, ksto, temp);
      out.push_back(temp); // k^2-1 rot res
    }


    for (int i = 0; i < 2; ++i) { //outer 4/2 kernel
      for (int j = 0; j < 9; ++j) { // inner k^2, 9 PCmult & 8 CCAdd
        ckks.context.PMult(out[j], m[i][j], temp);
        res[i] += temp;
      }
    }

    // rot op1 to op1
    // automorphism replace it with ccmult
    ckks.context.CCMult(op1, op1, temp, in_raise);
    DeviceVector after_modup = ckks.context.ModUp(in_raise);
    ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
    ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
    ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
    ckks.context.Add(temp, ksto, op1); // op1 to represent the ciphertext
    out.clear();
    out.push_back(op1); // record itself

    // 1-stage rot * 8 (autom -> modup -> innerp -> moddown --> add)
    for (int i = 0; i < 8; ++i) {
      // automorphism replace it with ccmult
      ckks.context.CCMult(op1, op1, temp, in_raise);
      DeviceVector after_modup = ckks.context.ModUp(in_raise);
      ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
      ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(temp, ksto, temp);
      out.push_back(temp); // k^2-1 rot res
    }

    for (int i = 0; i < 2; ++i) { //outer 4/2 kernel
      for (int j = 0; j < 9; ++j) { // inner k^2, 9 PCmult & 8 CCAdd
        ckks.context.PMult(out[j], m[i+2][j], temp);
        res[i] += temp;
      }
    }
    // Conv finish !!
  }

  void HE_Conv_3_Hoisting() {
    // ckks.context.is_modup_batched = true;
    // ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_; // init l
    auto key = ckks.GetRandomKey();
    Ciphertext op1 = ckks.GetRandomCiphertext(); // init ciphertext(ch1, ch2)
    vector<vector<Plaintext>> m(4, vector<Plaintext>(9));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 9; ++j) {
        m[i][j] = ckks.GetRandomPQPlaintext(); // PQ plaintext
      }
    }

    Ciphertext in_raise;
    Ciphertext d2o;
    Ciphertext ksto;
    vector<Ciphertext> out(9);
    Ciphertext temp;
    vector<Ciphertext> res(2);

    
    // 1-stage rot * 8 (autom -> modup -> innerp -> moddown --> add)
    out.push_back(ckks.GetRandomPQCiphertext()); // change op1 to PQ and record itself
    ckks.context.CCMult(op1, op1, temp, in_raise);
    // change temp to PQ
    temp = ckks.GetRandomPQCiphertext();
    DeviceVector after_modup = ckks.context.ModUp(in_raise);
    for (int i = 0; i < 8; ++i) {
      // automorphism replace it with ccmult
      ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
      // ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      // ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
      // (0, b') == temp
      ckks.context.Add(temp, d2o, temp);
      out.push_back(temp); // k^2-1 rot res
    }


    for (int i = 0; i < 2; ++i) { //outer 4/2 kernel
      for (int j = 0; j < 9; ++j) { // inner k^2, 9 PCmult & 8 CCAdd
        ckks.context.PMult(out[j], m[i][j], temp);
        res[i] += temp;
      }
    }

    // rot op1 to op1
    // automorphism replace it with ccmult
    ckks.context.CCMult(op1, op1, temp, in_raise);
    temp = ckks.GetRandomPQCiphertext();
    DeviceVector after_modup = ckks.context.ModUp(in_raise);
    ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
    ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
    ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
    ckks.context.Add(temp, ksto, op1); // op1 to represent the ciphertext
    out.clear();
    out.push_back(ckks.GetRandomPQCiphertext()); // change op1 to PQ and record itself

    // 1-stage rot * 8 (autom -> modup -> innerp -> moddown --> add)
    ckks.context.CCMult(op1, op1, temp, in_raise);
    DeviceVector after_modup = ckks.context.ModUp(in_raise);
    for (int i = 0; i < 8; ++i) {
      // automorphism replace it with ccmult
      ckks.context.KeySwitch(after_modup, key, d2o.ax__, d2o.bx__);
      // ckks.context.ModDown(d2o.ax__, ksto.ax__, num_moduli_after_moddown);
      // ckks.context.ModDown(d2o.bx__, ksto.bx__, num_moduli_after_moddown);
      ckks.context.Add(temp, d2o, temp);
      out.push_back(temp); // k^2-1 rot res
    }

    for (int i = 0; i < 2; ++i) { //outer 4/2 kernel
      for (int j = 0; j < 9; ++j) { // inner k^2, 9 PCmult & 8 CCAdd
        ckks.context.PMult(out[j], m[i+2][j], temp);
        res[i] += temp;
      }
    }

    // moddown
    vector<Ciphertext> ans(2);
    for (int i = 0; i <2; ++i) {
      ckks.context.ModDown(res[i], ans[i], num_moduli_after_moddown);
    }
    // Conv finish !!
  }

  void HE_Conv_3_My() {
    // ckks.context.is_modup_batched = true;
    // ckks.context.is_moddown_fused = true;
    ckks.context.is_keyswitch_fused = true;
    const int num_moduli_after_moddown = param.chain_length_; // init l
    auto key = ckks.GetRandomKey();
    vector<Ciphertext> input;
    for (int i = 0; i < param.dnum_; ++i) { // inital ciphertext index
      input.push_back(GetRandomACiphertext());
    }
    vector<vector<Plaintext>> m(4, vector<Plaintext>(9));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 9; ++j) {
        m[i][j] = ckks.GetRandomPQPlaintext(); // PQ plaintext
      }
    }
    DeviceVector vec = ckks.GetRandomPolyRNS(1);
    Ciphertext d2o;
    Ciphertext ksto;
    //vector<Ciphertext> out(9); // inner rot immediate result
    vector<vector<Ciphertext>> out(2, vector<Ciphertext>(9));
    Ciphertext temp;
    DeviceVector rax;
    DeviceVector rbx;
    vector<Ciphertext> res(2);
    auto in = ckks.GetRandomPolyAfterModUp(param.dnum_);
    MultPtxtBatch batcher(&ckks.context);

    for (int i = 0; i < param.dnum_; ++i) {
      temp.ax__.append(input[i].ax__);
      temp.bx__.append(input[i].bx__);
    }

    // 1-stage rot * 8 (autom -> modup -> innerp -> moddown --> add)
    // record itslef, unneccessary operate
    // for (int i = 0; i < param.dnum_; ++i) {
    //   out[0][i] = input[i];
    // }
    out[0][0] = temp;
    // rotate k^2 - 1
    for (int i = 0; i < param.dnum_; ++i) {
      ckks.context.AutomorphismTransform(input[i], rax, rbx, i, vec);
      auto raxo = ckks.context.ModUp(rax);
    }
    // in == append(raxo)
    for (int j = 0; j < 8; ++j) {
      ckks.context.KeySwitch(in, key, d2o.ax__, d2o.bx__);
      ckks.context.Add(d2o, d2o, d2o); // add
      out[0][j+1] = d2o; // restore
    }

    // rotate k^2 (1 out + k^2 - 1 rot)
    for (int i = 0; i < param.dnum_; ++i) {
      ckks.context.AutomorphismTransform(input[i], rax, rbx, i, vec);
      auto raxo = ckks.context.ModUp(rax);
    }
    for (int j = 0; j < 9; ++j) {
      ckks.context.KeySwitch(in, key, d2o.ax__, d2o.bx__);
      ckks.context.Add(d2o, d2o, d2o); // add
      out[1][j+1] = d2o; // restore
    }

    // Fusion MAC
    for (int i = 0; i < 9; ++i) {
      batcher.push2(out[0][i], m[0][i]);
      batcher.push2(out[1][i], m[2][i]);
    }
    batcher.flush2(res[0]);

    for (int i = 0; i < 9; ++i) {
      batcher.push2(out[0][i], m[1][i]);
      batcher.push2(out[1][i], m[3][i]);
    }
    batcher.flush2(res[1]);

    // moddown
    vector<Ciphertext> ans(2);
    for (int i = 0; i <2; ++i) {
      ckks.context.ModDown(res[i], ans[i], num_moduli_after_moddown);
    }
    // Conv finish !!
  }
/*
  void ModUpBench() {
    // std::cout <<param.degree_<< "\n" <<
    // param.level_<<"\n" <<
    // param.dnum_<<"\n" <<
    // param.alpha_<<"\n" <<
    // param.max_num_moduli_<<"\n" <<
    // param.chain_length_<<"\n" <<
    // param.num_special_moduli_<<std::endl;
    auto from = ckks.GetRandomPoly();
    ckks.context.is_modup_batched = false;
    // Run("ModUp", &Context::ModUp, from);
    ckks.context.is_modup_batched = true;
    Run("FusedModUp", &Context::ModUp, from);
  }

  void ModDownBench() {
    const int num_moduli_after_moddown = param.chain_length_;  // PQ -> Q
    auto from = ckks.GetRandomPolyRNS(param.max_num_moduli_);
    DeviceVector to;
    ckks.context.is_moddown_fused = false;
    // Run("ModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
    ckks.context.is_moddown_fused = true;
    Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
  }

  void KeyswitchBench() {
    auto key = ckks.GetRandomKey();
    auto in = ckks.GetRandomPolyAfterModUp(param.dnum_);  // beta = dnum case
    DeviceVector ax, bx;
    ckks.context.is_keyswitch_fused = false;
    // Run("KeySwitch", &Context::KeySwitch, in, key, ax, bx);
    ckks.context.is_keyswitch_fused = true;
    Run("FusedKeySwitch", &Context::KeySwitch, in, key, ax, bx);
  }
*/
  //  innerP
  void PtxtCtxtBatchBench() {
    int batch_size = 10;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext> op2(batch_size);
    // setup
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
    }
    auto MAD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum, out;
      ckks.context.PMult(op1[0], op2[0], accum);
      for (int i = 1; i < batch_size; i++) {
        ckks.context.PMult(op1[i], op2[i], out);
        ckks.context.Add(accum, out, accum);
      }
    };
    auto BatchMAD = [&](const auto& op1, const auto& op2) {
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
    };
    // Run("PtxtCtxtMAD", MAD, op1, op2);
    Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }

 private:
  Test ckks;
  Parameter param;
  int iters = 1;
};

int main() {
  // Benchmark bench(PARAM_LARGE_DNUM);
  Benchmark bench(new_stgcn1_1);
  return 0;
}