#include <jni.h>
#include <string>
#include <opencv2/core.hpp>

extern "C"
JNIEXPORT jstring JNICALL
Java_io_github_iostreamchik_scanner_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {

    std::string hello = "OpenCV version: " + cv::getVersionString();
    return env->NewStringUTF(hello.c_str());
}