package com.noguwo.dartllmlab.lfm3_bench

import android.os.Bundle
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity

class MainActivity : FlutterActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // The bench reads and writes the app's external files directory,
        // where adb can push and pull. Android gives the app that directory
        // only if the app asks for it, so ask before the Dart side starts.
        getExternalFilesDir(null)
        super.onCreate(savedInstanceState)
        // The benchmark runs unattended for minutes; a sleeping device would
        // pause it.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
}
