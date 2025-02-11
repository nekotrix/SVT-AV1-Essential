#
# Copyright(c) 2025 Alliance for Open Media
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

# Original option was USE_EXTERNAL_CPUINFO, which was removed in favor of USE_CPUINFO
# option(USE_EXTERNAL_CPUINFO "Consume system cpuinfo library only" OFF)

# Handle cpuinfo library
set(CPUINFO_DOCSTRING "Use pytorch's cpuinfo library. If COMPILE_C_ONLY is enabled, this is ignored")


if(DEFINED USE_EXTERNAL_CPUINFO OR DEFINED CACHE{USE_EXTERNAL_CPUINFO} AND USE_EXTERNAL_CPUINFO)
    set(USE_CPUINFO SYSTEM CACHE STRING ${CPUINFO_DOCSTRING} FORCE)
    unset(USE_EXTERNAL_CPUINFO CACHE)
    unset(USE_EXTERNAL_CPUINFO)
    message(WARNING "USE_EXTERNAL_CPUINFO is deprecated. Please use USE_CPUINFO=SYSTEM instead.")
else()
    set(USE_CPUINFO AUTO CACHE STRING ${CPUINFO_DOCSTRING})
endif()
# Off represents not using cpuinfo at all, disabling the use of simd where we don't have a fallback for.
# System represents looking for one provided externally such as through a package manager (using cmake's find_package).
#    Equivalent to previously setting `USE_EXTERNAL_CPUINFO` to `ON`. This will error out if not found.
# Local represents using the one provided in the third_party folder
# Auto represents first looking for a system one, and then falling back to the local one if not found
set_property(CACHE USE_CPUINFO PROPERTY STRINGS "OFF" "SYSTEM" "LOCAL" "AUTO")

set(HAVE_CPUINFO 0)
if(COMPILE_C_ONLY OR (NOT HAVE_X86_PLATFORM AND NOT HAVE_ARM_PLATFORM))
    set(USE_CPUINFO "OFF" CACHE STRING ${CPUINFO_DOCSTRING} FORCE)
    set(REQUESTED_CPUINFO "OFF")
else()
    string(TOUPPER "${USE_CPUINFO}" REQUESTED_CPUINFO)
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(cpuinfo_pc QUIET libcpuinfo)
endif()

if(REQUESTED_CPUINFO STREQUAL "SYSTEM")
    find_package(cpuinfo CONFIG REQUIRED)
elseif(REQUESTED_CPUINFO STREQUAL "AUTO")
    find_package(cpuinfo CONFIG QUIET)
endif()

if(REQUESTED_CPUINFO MATCHES "SYSTEM|AUTO" AND cpuinfo_FOUND)
    set(HAVE_CPUINFO 1)
    if(cpuinfo_pc_FOUND)
        string(APPEND REQUIRES_PRIVATE " libcpuinfo")
    endif()
    message(STATUS "Found system provided cpuinfo: ${cpuinfo_VERSION}")
endif()

if(REQUESTED_CPUINFO STREQUAL "LOCAL" OR (REQUESTED_CPUINFO STREQUAL "AUTO" AND NOT HAVE_CPUINFO))
    include(FetchContent)
    find_package(Git QUIET)
    if(GIT_EXECUTABLE)
        FetchContent_Declare(
            cpuinfo
            GIT_REPOSITORY https://github.com/1480c1/cpuinfo.git
            GIT_TAG e649baaa95efeb61517c06cc783287d4942ffe0e # Based on aaac07ee499895770c89163ce0920ef8bb41ed23, latest as of 2025-11-02
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/cpuinfo
            EXCLUDE_FROM_ALL
        )
    else()
        FetchContent_Declare(
            cpuinfo
            URL https://github.com/1480c1/cpuinfo/archive/e649baaa95efeb61517c06cc783287d4942ffe0e.zip
            URL_HASH SHA256=d2cfc3b4120ad2f17a7c6cc43d0547ecfb9f375b7918d723d234100822aa490e
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/cpuinfo
            EXCLUDE_FROM_ALL
        )
    endif()

    set(CPUINFO_LIBRARY_TYPE object CACHE STRING "")
    option(CPUINFO_BUILD_TOOLS "" OFF)
    option(CPUINFO_BUILD_UNIT_TESTS "" OFF)
    option(CPUINFO_BUILD_MOCK_TESTS "" OFF)
    option(CPUINFO_BUILD_BENCHMARKS "" OFF)
    option(CPUINFO_BUILD_PKG_CONFIG "" OFF)
    option(USE_SYSTEM_GOOGLEBENCHMARK "" OFF)
    option(USE_SYSTEM_GOOGLETEST "" OFF)
    FetchContent_MakeAvailable(cpuinfo)
    set(HAVE_CPUINFO 1)
    message(STATUS "cpuinfo not found and will be built locally")
endif()

if(REQUESTED_CPUINFO STREQUAL "OFF")
    add_library(cpuinfo INTERFACE)
    add_library(cpuinfo::cpuinfo ALIAS cpuinfo)
    message(STATUS "cpuinfo disabled")
endif()

add_compile_definitions(HAVE_CPUINFO=${HAVE_CPUINFO})
