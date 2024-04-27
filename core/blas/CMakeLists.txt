project(cncblas CUDA)

set(src_files
        "src/amaxmin.cu"
        "src/asum.cu"
        "src/axpy.cu"
)

add_library(${PROJECT_NAME} SHARED ${src_files})

target_include_directories(${PROJECT_NAME} PRIVATE include)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)