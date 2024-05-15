set(THIS cncblasTest)

set(TEST_SRC
        level-one/amaxTest.cu
        level-one/aminTest.cu
        level-one/asumTest.cu
        level-one/axpyTest.cu
        level-one/copyTest.cu
        level-one/dotTest.cu
        level-one/norm2Test.cu
        level-one/rotTest.cu
        level-one/scalTest.cu
        level-one/swapTest.cu
        level-two/gemvTest.cu
        level-two/gerTest.cu
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