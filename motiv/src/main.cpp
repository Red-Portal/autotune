
#include <vector>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <random>
#include <x86intrin.h>

#include "../../OpenBLAS/common.h"
#include <progresscpp/ProgressBar.hpp>
#include <nlohmann/json.hpp>

extern "C" 
{
  void blas_set_parameter();
  void openblas_read_env();
}

namespace chrono = std::chrono;
using millisec = chrono::duration<double, std::milli>;

void flush_vector(std::vector<double>& arr)
{
  for (size_t i = 0; i < arr.size(); i += 64/sizeof(double))
    _mm_clflush(&arr[i]);
}


int main()
{
  size_t iters = 64;

  char transa = 'N';
  char transb = 'N';

  double alpha[] = {1.0, 0.0};
  double beta[]  = {0.0, 0.0};

  auto b_seq = {1, 20, 40, 60, 80, 100};
  auto t_seq = {1, 2, 4, 8, 16, 32};

  auto m_seq = {4, 16, 64, 256, 1024, 4096};
  auto n_seq = {4, 16, 64, 256, 1024, 4096};
  auto k_seq = {4, 16, 64, 256, 1024, 4096};

  size_t total_its = m_seq.size()*n_seq.size()*k_seq.size()*b_seq.size()*t_seq.size();
  auto progbar     = progresscpp::ProgressBar(total_its, 50);

  auto log       = nlohmann::json();

  std::random_device seed{};
  auto rng   = std::mt19937(seed());
  auto dist  = std::normal_distribution<double>();
    
  for (auto block : b_seq)
  {
    if(setenv("OPENBLAS_BLOCK_FACTOR", std::to_string(block).c_str(), 1))
      throw std::runtime_error("setenv failed");

    for (auto thread : t_seq)
    {
      if(setenv("OPENBLAS_NUM_THREADS", std::to_string(thread).c_str(), 1))
	throw std::runtime_error("setenv failed");

			
      auto outstream = std::ofstream(
	"data_b" + std::to_string(block) + 
	"_t" + std::to_string(thread) + ".bson", 
	std::ios::out | std::ios::binary);

      openblas_read_env();
      blas_set_parameter();

      for (auto m : m_seq) {
	for (auto n : n_seq) {
	  for (auto k : k_seq) {
	    auto logentry = nlohmann::json();

	    auto a = std::vector<double>(m*k);
	    auto b = std::vector<double>(k*n);
	    auto c = std::vector<double>(m*n);

	    for (size_t i = 1; i < a.size(); ++i)
	      a[i] = dist(rng);
	    for (size_t i = 1; i < b.size(); ++i)
	      b[i] = dist(rng);
	    for (size_t i = 1; i < c.size(); ++i)
	      c[i] = dist(rng);

	    int lda = m;
	    int ldb = k;
	    int ldc = m;

	    auto measures = std::vector<double>(iters);
	    for (size_t t = 0; t < iters; ++t)
	    {
	      // flush cache
	      flush_vector(a);
	      flush_vector(b);
	      flush_vector(c);

	      auto start = chrono::steady_clock::now();
	      BLASFUNC(dgemm)(&transa, &transb, &m, &n, &k, alpha,
			      a.data(), &lda, b.data(), &ldb, beta, c.data(), &ldc);   
	      auto stop = chrono::steady_clock::now();
	      auto dur  = chrono::duration_cast<millisec>(stop - start).count();
	      measures[t] = dur;
	    }

	    ++progbar;
	    progbar.done();

	    logentry["block"]  = block;
	    logentry["thread"] = block;
	    logentry["m"]      = m;
	    logentry["n"]      = n;
	    logentry["k"]      = k;
	    logentry["time"]   = measures;

	    log.push_back(logentry);
	    auto top    = nlohmann::json();
	    top["data"] = log;
	    auto binary = nlohmann::json::to_bson(top);

	    outstream.seekp(0);
	    outstream.write(reinterpret_cast<char*>(binary.data()), binary.size());
	    outstream.flush();
	  }
	}
      }
      outstream.close();
    }
  }
}
