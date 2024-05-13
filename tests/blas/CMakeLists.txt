set(THIS cncblasTest)

set(TEST_SRC
        amaxTest.cu
        aminTest.cu
        asumTest.cu
        axpyTest.cu
)
add_executable(${THIS} ${TEST_SRC})
target_link_libraries(${THIS} PRIVATE
        gtest_main
        cncblas
        ${CUBLAS_LIBRARIES}
)
set_target_properties(${THIS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(
        NAME ${THIS}
        COMMAND ${THIS}
)