################################################################################
# Find packages
################################################################################
find_package(Eigen3 3.4 REQUIRED)

# OpenMP
if(OPENMP_ENABLED)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# PoseLib
include(FetchContent)
FetchContent_Declare(PoseLib
    GIT_REPOSITORY    https://github.com/PoseLib/PoseLib.git
    GIT_TAG           a84c545a9895e46d12a3f5ccde2581c25e6a6953
    EXCLUDE_FROM_ALL
)
message(STATUS "Configuring PoseLib...")
if (FETCH_POSELIB) 
    FetchContent_MakeAvailable(PoseLib)
else()
    find_package(PoseLib REQUIRED)
endif()
message(STATUS "Configuring PoseLib... done")

# COLMAP
find_package(COLMAP REQUIRED)

# Ceres
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changes the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()
if(INTERPOLATION_ENABLED)
    message(STATUS "Enabling pixelwise optimization with ceres interpolation. This should be disabled for clang.")
    add_definitions("-DINTERPOLATION_ENABLED")
else()
    message(STATUS "Disabling pixelwise optimization with ceres interpolation.")
endif()

