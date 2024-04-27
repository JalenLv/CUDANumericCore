set(sub_libs
        "blas"
)

foreach (sub_lib ${sub_libs})
  add_subdirectory(${sub_lib})
endforeach ()
