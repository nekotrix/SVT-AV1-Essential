set(codec_dir ${PROJECT_SOURCE_DIR}/Source/Lib/Codec)

add_executable(SvtScdEssential
    svt_scd_tool.c
    svt_scd_wrappers.h
    svt_scd_wrappers.c
    svt_scd_common.c
    svt_scd_shims.c
    ${codec_dir}/me_sad_calculation.c
    ${codec_dir}/pic_operators.c
    ${codec_dir}/svt_log.c
    ${codec_dir}/svt_threads.c
    ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/picture_operators_c.c)

target_include_directories(SvtScdEssential PRIVATE
    ${PROJECT_SOURCE_DIR}/Source/API
    ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT
    ${PROJECT_SOURCE_DIR}/Source/Lib/Globals
    ${codec_dir})
target_link_libraries(SvtScdEssential ${PLATFORM_LIBS})

if(NOT COMPILE_C_ONLY AND HAVE_X86_PLATFORM)
    add_library(svt_scd_avx2 OBJECT ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/transforms_intrin_avx2.c)
    target_include_directories(svt_scd_avx2 PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/API
        ${codec_dir}
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2)
    if(MSVC)
        target_compile_options(svt_scd_avx2 PRIVATE /arch:AVX2)
    else()
        target_compile_options(svt_scd_avx2 PRIVATE -mavx2)
    endif()
    target_compile_definitions(SvtScdEssential PRIVATE SVT_SCD_USE_AVX2=1)
    add_library(svt_scd_sse4_1 OBJECT ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE4_1/encodetxb_sse4.c)
    target_include_directories(svt_scd_sse4_1 PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/API
        ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2
        ${codec_dir})
    if(NOT MSVC)
        target_compile_options(svt_scd_sse4_1 PRIVATE -msse4.1)
    endif()
    target_sources(SvtScdEssential PRIVATE $<TARGET_OBJECTS:svt_scd_avx2> $<TARGET_OBJECTS:svt_scd_sse4_1>)
endif()

foreach(target SvtScdEssential svt_scd_avx2 svt_scd_sse4_1)
    if(TARGET ${target})
        if(MSVC)
            target_compile_options(${target} PRIVATE /Gy)
        else()
            target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
        endif()
    endif()
endforeach()

if(MSVC)
    target_link_options(SvtScdEssential PRIVATE /OPT:REF)
else()
    if(APPLE)
        target_link_options(SvtScdEssential PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(SvtScdEssential PRIVATE -Wl,--gc-sections)
    endif()
endif()

install(TARGETS SvtScdEssential RUNTIME COMPONENT Runtime DESTINATION ${CMAKE_INSTALL_BINDIR})
