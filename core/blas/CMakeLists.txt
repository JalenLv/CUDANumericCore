project(cncblas CUDA)

set(src_files
        src/amaxmin.cu
        src/asum.cu
        src/axpy.cu
        src/copy.cu
        src/dot.cu
        src/norm2.cu
        src/rot.cu
        src/scal.cu
        src/swap.cu
)

add_library(${PROJECT_NAME} SHARED ${src_files})

target_include_directories(${PROJECT_NAME} PRIVATE
        include
        ../../common
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
