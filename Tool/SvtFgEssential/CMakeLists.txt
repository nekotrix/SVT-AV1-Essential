set(codec_dir ${PROJECT_SOURCE_DIR}/Source/Lib/Codec)
set(c_default_dir ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT)

add_executable(SvtFgEssential
    svt_fg_tool.c
    svt_fg_shims.c
    svt_fg_common.c
    svt_fg_noise_model.c
    svt_fg_noise_model.h
    ${codec_dir}/noise_util.c
    ${codec_dir}/fft.c
    ${codec_dir}/grainSynthesis.c
    ${codec_dir}/pic_operators.c
    ${codec_dir}/svt_log.c
    ${codec_dir}/svt_malloc.c
    ${codec_dir}/svt_threads.c
    ${c_default_dir}/pack_unpack_c.c
    ${c_default_dir}/picture_operators_c.c)

target_include_directories(SvtFgEssential PRIVATE
    ${PROJECT_SOURCE_DIR}/Source/API
    ${codec_dir}
    ${c_default_dir})
# Retain the C fallbacks used by the tool's dispatch table.
target_compile_definitions(SvtFgEssential PRIVATE
    CONFIG_ENABLE_FILM_GRAIN=1
    SVT_AV1_UNIT_TEST_BUILD=1)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(SvtFgEssential PRIVATE _GNU_SOURCE)
endif()
target_link_libraries(SvtFgEssential ${PLATFORM_LIBS})

if(NOT COMPILE_C_ONLY AND HAVE_X86_PLATFORM)
    add_library(svt_fg_avx2 OBJECT
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2/fft_sse2.c
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/fft_avx2.c
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/noise_model_avx2.c
        svt_fg_fast_avx2.c)
    target_include_directories(svt_fg_avx2 PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/API
        ${codec_dir}
        ${c_default_dir})
    target_compile_definitions(svt_fg_avx2 PRIVATE CONFIG_ENABLE_FILM_GRAIN=1)
    if(MSVC)
        target_compile_options(svt_fg_avx2 PRIVATE /arch:AVX2)
    else()
        target_compile_options(svt_fg_avx2 PRIVATE -mavx2)
    endif()
    target_compile_definitions(SvtFgEssential PRIVATE SVT_FG_USE_AVX2=1)
    target_sources(SvtFgEssential PRIVATE $<TARGET_OBJECTS:svt_fg_avx2>)
endif()

if(MSVC)
    target_compile_options(SvtFgEssential PRIVATE /Gy)
    target_link_options(SvtFgEssential PRIVATE /OPT:REF)
else()
    target_compile_options(SvtFgEssential PRIVATE -ffunction-sections -fdata-sections)
    if(APPLE)
        target_link_options(SvtFgEssential PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(SvtFgEssential PRIVATE -Wl,--gc-sections)
    endif()
endif()

install(TARGETS SvtFgEssential RUNTIME COMPONENT Runtime DESTINATION ${CMAKE_INSTALL_BINDIR})
