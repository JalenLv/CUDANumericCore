import os
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def new_log(functions, durations, categories, logFilePath, N):
  subprocess.run("rm -f {}/*".format(logFilePath), shell=True)
  print("Original log files cleaned.")

  for i in range(N):
    print("Iteration {} started.".format(i + 1))
    
    subprocess.run(
      'ncu ./build/tests/blas/cncblasTest | grep -E "(CC)|(Duration)" | cat > {}/log_{}.log'.format(
        logFilePath,
        i + 1,
      ),
      shell=True,
    )
    print('Ncu executed for iteration {}.'.format(i + 1))

    with open("{}/log_{}.log".format(logFilePath, i + 1), "r") as file:
      lines = file.readlines()
    print("Log file read for iteration {}.".format(i + 1))

    # level-one
    # amax
    functions.append("Samax")
    duration = float(lines[1].split()[-1]) + float(lines[3].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Samax")
    duration = float(lines[5].split()[-1]) + float(lines[7].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Damax")
    duration = float(lines[9].split()[-1]) + float(lines[11].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Damax")
    duration = float(lines[13].split()[-1]) + float(lines[15].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Camax")
    duration = float(lines[17].split()[-1]) + float(lines[19].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Camax")
    duration = float(lines[21].split()[-1]) + float(lines[23].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Zamax")
    duration = float(lines[25].split()[-1]) + float(lines[27].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Zamax")
    duration = float(lines[29].split()[-1]) + float(lines[31].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # amin
    functions.append("Samin")
    duration = float(lines[33].split()[-1]) + float(lines[35].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Samin")
    duration = float(lines[37].split()[-1]) + float(lines[39].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Damin")
    duration = float(lines[41].split()[-1]) + float(lines[43].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Damin")
    duration = float(lines[45].split()[-1]) + float(lines[47].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Camin")
    duration = float(lines[49].split()[-1]) + float(lines[51].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Camin")
    duration = float(lines[53].split()[-1]) + float(lines[55].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zamin")
    duration = float(lines[57].split()[-1]) + float(lines[59].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Zamin")
    duration = float(lines[61].split()[-1]) + float(lines[63].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # asum
    functions.append("Sasum")
    duration = float(lines[65].split()[-1]) + float(lines[67].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sasum")
    duration = float(lines[69].split()[-1]) + float(lines[71].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dasum")
    duration = float(lines[73].split()[-1]) + float(lines[75].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dasum")
    duration = float(lines[77].split()[-1]) + float(lines[79].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Casum")
    duration = float(lines[81].split()[-1]) + float(lines[83].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Casum")
    duration = float(lines[85].split()[-1]) + float(lines[87].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zasum")
    duration = float(lines[89].split()[-1]) + float(lines[91].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zasum")
    duration = float(lines[93].split()[-1]) + float(lines[95].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # axpy
    functions.append("Saxpy")
    duration = float(lines[97].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Saxpy")
    duration = float(lines[99].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Daxpy")
    duration = float(lines[101].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Daxpy")
    duration = float(lines[103].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Caxpy")
    duration = float(lines[105].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Caxpy")
    duration = float(lines[107].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zaxpy")
    duration = float(lines[109].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zaxpy")
    duration = float(lines[111].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # copy
    functions.append("Scopy")
    duration = float(lines[113].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dcopy")
    duration = float(lines[115].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Ccopy")
    duration = float(lines[117].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zcopy")
    duration = float(lines[119].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    # dot
    functions.append("Sdot")
    duration = float(lines[121].split()[-1]) + float(lines[123].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sdot")
    duration = float(lines[125].split()[-1]) + float(lines[127].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Ddot")
    duration = float(lines[129].split()[-1]) + float(lines[131].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Ddot")
    duration = float(lines[133].split()[-1]) + float(lines[135].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cdotu")
    duration = float(lines[137].split()[-1]) + float(lines[139].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cdotu")
    duration = float(lines[141].split()[-1]) + float(lines[143].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cdotc")
    duration = float(lines[145].split()[-1]) + float(lines[147].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cdotc")
    duration = float(lines[149].split()[-1]) + float(lines[151].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdotu")
    duration = float(lines[153].split()[-1]) + float(lines[155].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdotu")
    duration = float(lines[157].split()[-1]) + float(lines[159].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdotc")
    duration = float(lines[161].split()[-1]) + float(lines[163].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdotc")
    duration = float(lines[165].split()[-1]) + float(lines[167].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # nrm2
    functions.append("Snrm2")
    duration = float(lines[169].split()[-1]) + float(lines[171].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Snrm2")
    duration = float(lines[173].split()[-1]) + float(lines[175].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dnrm2")
    duration = float(lines[177].split()[-1]) + float(lines[179].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dnrm2")
    duration = float(lines[181].split()[-1]) + float(lines[183].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cnrm2")
    duration = float(lines[185].split()[-1]) + float(lines[187].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cnrm2")
    duration = float(lines[189].split()[-1]) + float(lines[191].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Znrm2")
    duration = float(lines[193].split()[-1]) + float(lines[195].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Znrm2")
    duration = float(lines[197].split()[-1]) + float(lines[199].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # rot
    functions.append("Srot")
    duration = float(lines[201].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Srot")
    duration = float(lines[203].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Drot")
    duration = float(lines[205].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Drot")
    duration = float(lines[207].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # scal
    functions.append("Sscal")
    duration = float(lines[209].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sscal")
    duration = float(lines[211].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dscal")
    duration = float(lines[213].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dscal")
    duration = float(lines[215].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cscal")
    duration = float(lines[217].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cscal")
    duration = float(lines[219].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zscal")
    duration = float(lines[221].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zscal")
    duration = float(lines[223].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Csscal")
    duration = float(lines[225].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Csscal")
    duration = float(lines[227].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdscal")
    duration = float(lines[229].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdscal")
    duration = float(lines[231].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # swap
    functions.append("Sswap")
    duration = float(lines[233].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sswap")
    duration = float(lines[235].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dswap")
    duration = float(lines[237].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dswap")
    duration = float(lines[239].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cswap")
    duration = float(lines[241].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cswap")
    duration = float(lines[243].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zswap")
    duration = float(lines[245].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zswap")
    duration = float(lines[247].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # level-two
    # gbmv
    functions.append("Sgbmv_N")
    duration = float(lines[249].split()[-1]) + float(lines[251].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Sgbmv_T")
    duration = float(lines[257].split()[-1]) + float(lines[259].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Dgbmv_N")
    duration = float(lines[265].split()[-1]) + float(lines[267].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgbmv_T")
    duration = float(lines[273].split()[-1]) + float(lines[275].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgbmv_N")
    duration = float(lines[281].split()[-1]) + float(lines[283].split()[-1]) + float(lines[285].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgbmv_T")
    duration = float(lines[291].split()[-1]) + float(lines[293].split()[-1]) + float(lines[295].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Cgbmv_C")
    duration = float(lines[303].split()[-1]) + float(lines[305].split()[-1]) + float(lines[307].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_N")
    duration = float(lines[315].split()[-1]) + float(lines[317].split()[-1]) + float(lines[319].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_T")
    duration = float(lines[325].split()[-1]) + float(lines[327].split()[-1]) + float(lines[329].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_C")
    duration = float(lines[337].split()[-1]) + float(lines[339].split()[-1]) + float(lines[341].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # gemv
    functions.append("Sgemv_N")
    duration = float(lines[349].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Sgemv_N")
    duration = float(lines[351].split()[-1]) + float(lines[353].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgemv_N")
    duration = float(lines[355].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dgemv_N")
    duration = float(lines[357].split()[-1]) + float(lines[359].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_N")
    duration = float(lines[361].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_N")
    duration = float(lines[363].split()[-1]) + float(lines[365].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_N")
    duration = float(lines[367].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_N")
    duration = float(lines[369].split()[-1]) + float(lines[371].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Sgemv_T")
    duration = float(lines[373].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sgemv_T")
    duration = float(lines[375].split()[-1]) + float(lines[377].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgemv_T")
    duration = float(lines[379].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dgemv_T")
    duration = float(lines[381].split()[-1]) + float(lines[383].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_T")
    duration = float(lines[385].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_T")
    duration = float(lines[387].split()[-1]) + float(lines[389].split()[-1]) + float(lines[391].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_T")
    duration = float(lines[393].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_T")
    duration = float(lines[395].split()[-1]) + float(lines[397].split()[-1]) + float(lines[399].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_C")
    duration = float(lines[401].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_C")
    duration = float(lines[403].split()[-1]) + float(lines[405].split()[-1]) + float(lines[407].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_C")
    duration = float(lines[409].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_C")
    duration = float(lines[411].split()[-1]) + float(lines[413].split()[-1]) + float(lines[415].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # ger
    functions.append("Sger")
    duration = float(lines[417].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sger")
    duration = float(lines[419].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dger")
    duration = float(lines[421].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dger")
    duration = float(lines[423].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgeru")
    duration = float(lines[425].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgeru")
    duration = float(lines[427].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgeru")
    duration = float(lines[429].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgeru")
    duration = float(lines[431].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgerc")
    duration = float(lines[433].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgerc")
    duration = float(lines[435].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgerc")
    duration = float(lines[437].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgerc")
    duration = float(lines[439].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    print('Data extracted for iteration {}.'.format(i + 1))
    print("Iteration {} done.".format(i + 1))
    print()

def read_log(functions, durations, catefories, logFilePath, N):
  for i in range(N):
    print("Iteration {} started.".format(i + 1))
    
    with open("{}/log_{}.log".format(logFilePath, i + 1), "r") as file:
      lines = file.readlines()
    print("Log file read for iteration {}.".format(i + 1))

    # level-one
    # amax
    functions.append("Samax")
    duration = float(lines[1].split()[-1]) + float(lines[3].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Samax")
    duration = float(lines[5].split()[-1]) + float(lines[7].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Damax")
    duration = float(lines[9].split()[-1]) + float(lines[11].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Damax")
    duration = float(lines[13].split()[-1]) + float(lines[15].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Camax")
    duration = float(lines[17].split()[-1]) + float(lines[19].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Camax")
    duration = float(lines[21].split()[-1]) + float(lines[23].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Zamax")
    duration = float(lines[25].split()[-1]) + float(lines[27].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Zamax")
    duration = float(lines[29].split()[-1]) + float(lines[31].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # amin
    functions.append("Samin")
    duration = float(lines[33].split()[-1]) + float(lines[35].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Samin")
    duration = float(lines[37].split()[-1]) + float(lines[39].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Damin")
    duration = float(lines[41].split()[-1]) + float(lines[43].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Damin")
    duration = float(lines[45].split()[-1]) + float(lines[47].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Camin")
    duration = float(lines[49].split()[-1]) + float(lines[51].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Camin")
    duration = float(lines[53].split()[-1]) + float(lines[55].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zamin")
    duration = float(lines[57].split()[-1]) + float(lines[59].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Zamin")
    duration = float(lines[61].split()[-1]) + float(lines[63].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # asum
    functions.append("Sasum")
    duration = float(lines[65].split()[-1]) + float(lines[67].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sasum")
    duration = float(lines[69].split()[-1]) + float(lines[71].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dasum")
    duration = float(lines[73].split()[-1]) + float(lines[75].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dasum")
    duration = float(lines[77].split()[-1]) + float(lines[79].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Casum")
    duration = float(lines[81].split()[-1]) + float(lines[83].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Casum")
    duration = float(lines[85].split()[-1]) + float(lines[87].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zasum")
    duration = float(lines[89].split()[-1]) + float(lines[91].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zasum")
    duration = float(lines[93].split()[-1]) + float(lines[95].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # axpy
    functions.append("Saxpy")
    duration = float(lines[97].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Saxpy")
    duration = float(lines[99].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Daxpy")
    duration = float(lines[101].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Daxpy")
    duration = float(lines[103].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Caxpy")
    duration = float(lines[105].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Caxpy")
    duration = float(lines[107].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zaxpy")
    duration = float(lines[109].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zaxpy")
    duration = float(lines[111].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # copy
    functions.append("Scopy")
    duration = float(lines[113].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dcopy")
    duration = float(lines[115].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Ccopy")
    duration = float(lines[117].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zcopy")
    duration = float(lines[119].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    # dot
    functions.append("Sdot")
    duration = float(lines[121].split()[-1]) + float(lines[123].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sdot")
    duration = float(lines[125].split()[-1]) + float(lines[127].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Ddot")
    duration = float(lines[129].split()[-1]) + float(lines[131].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Ddot")
    duration = float(lines[133].split()[-1]) + float(lines[135].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cdotu")
    duration = float(lines[137].split()[-1]) + float(lines[139].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cdotu")
    duration = float(lines[141].split()[-1]) + float(lines[143].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cdotc")
    duration = float(lines[145].split()[-1]) + float(lines[147].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cdotc")
    duration = float(lines[149].split()[-1]) + float(lines[151].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdotu")
    duration = float(lines[153].split()[-1]) + float(lines[155].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdotu")
    duration = float(lines[157].split()[-1]) + float(lines[159].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdotc")
    duration = float(lines[161].split()[-1]) + float(lines[163].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdotc")
    duration = float(lines[165].split()[-1]) + float(lines[167].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # nrm2
    functions.append("Snrm2")
    duration = float(lines[169].split()[-1]) + float(lines[171].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Snrm2")
    duration = float(lines[173].split()[-1]) + float(lines[175].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dnrm2")
    duration = float(lines[177].split()[-1]) + float(lines[179].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dnrm2")
    duration = float(lines[181].split()[-1]) + float(lines[183].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cnrm2")
    duration = float(lines[185].split()[-1]) + float(lines[187].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cnrm2")
    duration = float(lines[189].split()[-1]) + float(lines[191].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Znrm2")
    duration = float(lines[193].split()[-1]) + float(lines[195].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Znrm2")
    duration = float(lines[197].split()[-1]) + float(lines[199].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # rot
    functions.append("Srot")
    duration = float(lines[201].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Srot")
    duration = float(lines[203].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Drot")
    duration = float(lines[205].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Drot")
    duration = float(lines[207].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # scal
    functions.append("Sscal")
    duration = float(lines[209].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sscal")
    duration = float(lines[211].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dscal")
    duration = float(lines[213].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dscal")
    duration = float(lines[215].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cscal")
    duration = float(lines[217].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cscal")
    duration = float(lines[219].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zscal")
    duration = float(lines[221].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zscal")
    duration = float(lines[223].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Csscal")
    duration = float(lines[225].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Csscal")
    duration = float(lines[227].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zdscal")
    duration = float(lines[229].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zdscal")
    duration = float(lines[231].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # swap
    functions.append("Sswap")
    duration = float(lines[233].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sswap")
    duration = float(lines[235].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dswap")
    duration = float(lines[237].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dswap")
    duration = float(lines[239].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cswap")
    duration = float(lines[241].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cswap")
    duration = float(lines[243].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zswap")
    duration = float(lines[245].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zswap")
    duration = float(lines[247].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # level-two
    # gbmv
    functions.append("Sgbmv_N")
    duration = float(lines[249].split()[-1]) + float(lines[251].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Sgbmv_T")
    duration = float(lines[257].split()[-1]) + float(lines[259].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Dgbmv_N")
    duration = float(lines[265].split()[-1]) + float(lines[267].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgbmv_T")
    duration = float(lines[273].split()[-1]) + float(lines[275].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgbmv_N")
    duration = float(lines[281].split()[-1]) + float(lines[283].split()[-1]) + float(lines[285].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgbmv_T")
    duration = float(lines[291].split()[-1]) + float(lines[293].split()[-1]) + float(lines[295].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    functions.append("Cgbmv_C")
    duration = float(lines[303].split()[-1]) + float(lines[305].split()[-1]) + float(lines[307].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_N")
    duration = float(lines[315].split()[-1]) + float(lines[317].split()[-1]) + float(lines[319].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_T")
    duration = float(lines[325].split()[-1]) + float(lines[327].split()[-1]) + float(lines[329].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgbmv_C")
    duration = float(lines[337].split()[-1]) + float(lines[339].split()[-1]) + float(lines[341].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")

    # gemv
    functions.append("Sgemv_N")
    duration = float(lines[349].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")

    functions.append("Sgemv_N")
    duration = float(lines[351].split()[-1]) + float(lines[353].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgemv_N")
    duration = float(lines[355].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dgemv_N")
    duration = float(lines[357].split()[-1]) + float(lines[359].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_N")
    duration = float(lines[361].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_N")
    duration = float(lines[363].split()[-1]) + float(lines[365].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_N")
    duration = float(lines[367].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_N")
    duration = float(lines[369].split()[-1]) + float(lines[371].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Sgemv_T")
    duration = float(lines[373].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sgemv_T")
    duration = float(lines[375].split()[-1]) + float(lines[377].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dgemv_T")
    duration = float(lines[379].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dgemv_T")
    duration = float(lines[381].split()[-1]) + float(lines[383].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_T")
    duration = float(lines[385].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_T")
    duration = float(lines[387].split()[-1]) + float(lines[389].split()[-1]) + float(lines[391].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_T")
    duration = float(lines[393].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_T")
    duration = float(lines[395].split()[-1]) + float(lines[397].split()[-1]) + float(lines[399].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgemv_C")
    duration = float(lines[401].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgemv_C")
    duration = float(lines[403].split()[-1]) + float(lines[405].split()[-1]) + float(lines[407].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgemv_C")
    duration = float(lines[409].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgemv_C")
    duration = float(lines[411].split()[-1]) + float(lines[413].split()[-1]) + float(lines[415].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    # ger
    functions.append("Sger")
    duration = float(lines[417].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Sger")
    duration = float(lines[419].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Dger")
    duration = float(lines[421].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Dger")
    duration = float(lines[423].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgeru")
    duration = float(lines[425].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgeru")
    duration = float(lines[427].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgeru")
    duration = float(lines[429].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgeru")
    duration = float(lines[431].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Cgerc")
    duration = float(lines[433].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Cgerc")
    duration = float(lines[435].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    functions.append("Zgerc")
    duration = float(lines[437].split()[-1])
    durations.append(duration)
    categories.append("cuBLAS")
    
    functions.append("Zgerc")
    duration = float(lines[439].split()[-1])
    durations.append(duration)
    categories.append("cncBLAS")
    
    print('Data extracted for iteration {}.'.format(i + 1))
    print("Iteration {} done.".format(i + 1))
    print()
  

N = 200
logFilePath = "./ncu_log"

functions = []
durations = []
categories = []

# new_log(functions, durations, categories, logFilePath, N)
read_log(functions, durations, categories, logFilePath, N)

data = {"Function": functions, "Duration": durations, "Category": categories}
df = pd.DataFrame(data)
df_grouped = df.groupby(["Category", "Function"]).mean().reset_index()
df_pivot = df_grouped.pivot(index="Function", columns="Category", values="Duration")

print(df)
print(df_grouped)
print(df_pivot)

df_pivot.plot(kind="bar", figsize=(20, 10))

plt.figure(figsize=(20, 10))
sns.barplot(data=df_grouped, x="Function", y="Duration", hue="Category")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
