include(FetchContent)

FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP on
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip

)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)
ADD_EXECUTABLE(FusionTest
  public/FusionTest.cpp
)

TARGET_LINK_LIBRARIES(FusionTest PRIVATE DeviceFunctions gtest_main)
