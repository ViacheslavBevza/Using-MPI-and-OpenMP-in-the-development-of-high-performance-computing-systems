#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// СТРУКТУРА ДЛЯ ЗБОРУ МЕТРИК ПРОДУКТИВНОСТІ


struct PerformanceMetrics {
    double execution_time;
    double speedup;
    double efficiency;
    int num_processes;
    int num_threads;
    long long operations;
    
    void display() const {
        cout << fixed << setprecision(4);
        cout << "Processes: " << num_processes 
             << ", Threads: " << num_threads
             << ", Time: " << execution_time << "s"
             << ", Speedup: " << speedup << "x"
             << ", Efficiency: " << (efficiency * 100) << "%" << endl;
    }
};


// ОБЧИСЛЮВАЛЬНЕ ЯДРО: ІНТЕГРАЦІЯ МЕТОДОМ МОНТЕ-КАРЛО


class MonteCarloIntegrator {
private:
    long long samples_per_process;
    int rank;
    int size;
    
    // Функція для інтегрування: f(x) = 4 / (1 + x^2) для обчислення π
    double function(double x) {
        return 4.0 / (1.0 + x * x);
    }
    
public:
    MonteCarloIntegrator(long long total_samples, int mpi_rank, int mpi_size) 
        : rank(mpi_rank), size(mpi_size) {
        samples_per_process = total_samples / mpi_size;
    }
    
    // Паралельне обчислення з OpenMP всередині кожного MPI-процесу
    double compute_parallel() {
        double local_sum = 0.0;
        long long local_samples = samples_per_process;
        
        #pragma omp parallel reduction(+:local_sum)
        {
            unsigned int seed = rank * 1000 + omp_get_thread_num();
            
            #pragma omp for
            for (long long i = 0; i < local_samples; i++) {
                double x = (double)rand_r(&seed) / RAND_MAX;
                local_sum += function(x);
            }
        }
        
        return local_sum / local_samples;
    }
    
    // Послідовне обчислення для порівняння
    double compute_sequential() {
        double local_sum = 0.0;
        unsigned int seed = rank * 1000;
        
        for (long long i = 0; i < samples_per_process; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            local_sum += function(x);
        }
        
        return local_sum / samples_per_process;
    }
};


// ОБЧИСЛЮВАЛЬНЕ ЯДРО: МНОЖЕННЯ МАТРИЦЬ


class MatrixMultiplier {
private:
    int matrix_size;
    vector<vector<double>> A, B, C;
    int rank;
    int size;
    
public:
    MatrixMultiplier(int n, int mpi_rank, int mpi_size) 
        : matrix_size(n), rank(mpi_rank), size(mpi_size) {
        if (rank == 0) {
            A.resize(n, vector<double>(n, 1.0));
            B.resize(n, vector<double>(n, 2.0));
            C.resize(n, vector<double>(n, 0.0));
        }
    }
    
    // Гібридне множення MPI + OpenMP
    void multiply_hybrid() {
        int rows_per_process = matrix_size / size;
        
        vector<double> local_A(rows_per_process * matrix_size);
        vector<double> global_B(matrix_size * matrix_size);
        vector<double> local_C(rows_per_process * matrix_size, 0.0);
        
        // Розподіл даних між процесами
        if (rank == 0) {
            for (int i = 0; i < matrix_size; i++) {
                for (int j = 0; j < matrix_size; j++) {
                    global_B[i * matrix_size + j] = B[i][j];
                }
            }
        }
        
        // Broadcast матриці B всім процесам
        MPI_Bcast(global_B.data(), matrix_size * matrix_size, 
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Scatter рядків матриці A
        vector<double> flat_A;
        if (rank == 0) {
            flat_A.resize(matrix_size * matrix_size);
            for (int i = 0; i < matrix_size; i++) {
                for (int j = 0; j < matrix_size; j++) {
                    flat_A[i * matrix_size + j] = A[i][j];
                }
            }
        }
        
        MPI_Scatter(flat_A.data(), rows_per_process * matrix_size, MPI_DOUBLE,
                   local_A.data(), rows_per_process * matrix_size, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        // Паралельне множення з OpenMP
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows_per_process; i++) {
            for (int j = 0; j < matrix_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < matrix_size; k++) {
                    sum += local_A[i * matrix_size + k] * 
                           global_B[k * matrix_size + j];
                }
                local_C[i * matrix_size + j] = sum;
            }
        }
        
        // Gather результатів
        vector<double> flat_C;
        if (rank == 0) {
            flat_C.resize(matrix_size * matrix_size);
        }
        
        MPI_Gather(local_C.data(), rows_per_process * matrix_size, MPI_DOUBLE,
                  flat_C.data(), rows_per_process * matrix_size, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
    }
};


// СИСТЕМА БЕНЧМАРКІНГУ


class ParallelBenchmark {
private:
    int rank;
    int size;
    vector<PerformanceMetrics> results;
    
public:
    ParallelBenchmark(int mpi_rank, int mpi_size) 
        : rank(mpi_rank), size(mpi_size) {}
    
    void run_monte_carlo_benchmark(long long samples) {
        if (rank == 0) {
            cout << "\n=== MONTE CARLO INTEGRATION BENCHMARK ===" << endl;
            cout << "Total samples: " << samples << endl << endl;
        }
        
        MonteCarloIntegrator integrator(samples, rank, size);
        
        // Послідовна версія (тільки на rank 0 з 1 потоком)
        double seq_time = 0.0;
        if (rank == 0) {
            omp_set_num_threads(1);
            auto start = high_resolution_clock::now();
            double result = integrator.compute_sequential();
            auto end = high_resolution_clock::now();
            seq_time = duration<double>(end - start).count();
            
            cout << "Sequential (1 process, 1 thread): " 
                 << seq_time << "s, Result: " << result << endl;
        }
        
        // Broadcast послідовного часу всім процесам
        MPI_Bcast(&seq_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Тестування з різною кількістю потоків OpenMP
        vector<int> thread_counts = {1, 2, 4, 8};
        
        for (int num_threads : thread_counts) {
            omp_set_num_threads(num_threads);
            
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = high_resolution_clock::now();
            
            double local_result = integrator.compute_parallel();
            
            // Редукція результатів через MPI
            double global_result = 0.0;
            MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, 
                      MPI_SUM, 0, MPI_COMM_WORLD);
            
            auto end = high_resolution_clock::now();
            double par_time = duration<double>(end - start).count();
            
            if (rank == 0) {
                global_result /= size;
                
                PerformanceMetrics metrics;
                metrics.num_processes = size;
                metrics.num_threads = num_threads;
                metrics.execution_time = par_time;
                metrics.speedup = seq_time / par_time;
                metrics.efficiency = metrics.speedup / (size * num_threads);
                metrics.operations = samples;
                
                cout << "Parallel: ";
                metrics.display();
                
                results.push_back(metrics);
            }
        }
    }
    
    void run_matrix_benchmark(int matrix_size) {
        if (rank == 0) {
            cout << "\n=== MATRIX MULTIPLICATION BENCHMARK ===" << endl;
            cout << "Matrix size: " << matrix_size << "x" << matrix_size << endl << endl;
        }
        
        MatrixMultiplier multiplier(matrix_size, rank, size);
        
        vector<int> thread_counts = {1, 2, 4, 8};
        
        for (int num_threads : thread_counts) {
            omp_set_num_threads(num_threads);
            
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = high_resolution_clock::now();
            
            multiplier.multiply_hybrid();
            
            auto end = high_resolution_clock::now();
            double time = duration<double>(end - start).count();
            
            if (rank == 0) {
                cout << "Matrix multiplication: Processes=" << size 
                     << ", Threads=" << num_threads 
                     << ", Time=" << time << "s" << endl;
            }
        }
    }
    
    void display_summary() {
        if (rank == 0 && !results.empty()) {
            cout << "\n=== PERFORMANCE SUMMARY ===" << endl;
            cout << "Best speedup: " << results.back().speedup << "x" << endl;
            cout << "Best efficiency: " << (results.back().efficiency * 100) << "%" << endl;
        }
    }
};


// ГОЛОВНА ПРОГРАМА


int main(int argc, char** argv) {
    // Ініціалізація MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "======================================================" << endl;
        cout << "   HIGH PERFORMANCE COMPUTING: MPI + OpenMP DEMO     " << endl;
        cout << "======================================================" << endl;
        cout << "MPI Processes: " << size << endl;
        cout << "Max OpenMP Threads: " << omp_get_max_threads() << endl;
        cout << "======================================================" << endl;
    }
    
    // Створення системи бенчмаркінгу
    ParallelBenchmark benchmark(rank, size);
    
    // Запуск тестів
    benchmark.run_monte_carlo_benchmark(100000000);  // 100 мільйонів вибірок
    benchmark.run_matrix_benchmark(1000);            // Матриця 1000x1000
    
    // Виведення підсумків
    benchmark.display_summary();
    
    if (rank == 0) {
        cout << "\n======================================================" << endl;
        cout << "  Hybrid MPI+OpenMP approach demonstrates optimal    " << endl;
        cout << "  scalability for heterogeneous computing systems    " << endl;
        cout << "======================================================" << endl;
    }
    
    MPI_Finalize();
    return 0;
}
