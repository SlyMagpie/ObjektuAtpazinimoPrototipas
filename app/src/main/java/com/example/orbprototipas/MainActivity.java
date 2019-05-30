package com.example.orbprototipas;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final int REQUEST_PERMISSION = 100;
    private int width, height;
    private CameraBridgeViewBase mOpenCvCameraView;

    Button modeButton;
    Button protoDisplayButton;

    Scalar RED = new Scalar(255, 0, 0); //ORB
    Scalar GREEN = new Scalar(0, 255, 0); //AKAZE
    Scalar BLUE = new Scalar(0, 0, 255); //BRISK

    ORB detectorORB;
    BRISK detectorBRISK;
    AKAZE detectorAKAZE;

    DescriptorMatcher matcher;

    private double minDist = 40; //Best results with ORB. To get better results with other methods, test and change this value.

    private int currentProtoFrame = 0;
    private int protoDisplayMode = 0; //[ORB, AKAZE, BRISK] for display purposes only
    private int statisticsLimit = 240; //The equivalent of 10 seconds of filming at 24 fps
    private long[] detectionORB, detectionAKAZE, detectionBRISK; //Keypoint detection time
    private long[] descriptionORB, descriptionAKAZE, descriptionBRISK; //Time of descriptor creation from detected keypoints
    private long[] matchingORB, matchingAKAZE, matchingBRISK; //Time of matching the descriptors of a frame to an image
    private double avgMatchesORB = 0, avgMatchesAKAZE = 0, avgMatchesBRISK = 0; //Total number of matches, divided by statisticsLimit to get averages

    //Prototype detectors and descriptors for testing most available up-to-date methods
    Mat protoDescriptorsORB, protoDescriptorsAKAZE, protoDescriptorsBRISK;
    MatOfKeyPoint protoKeypointsORB, protoKeypointsAKAZE, protoKeypointsBRISK;
    Mat protoImage;

    //Arrays used in feature recognition by ORB in secondary mode
    Mat[] multiImageArray;
    Mat[] multiImageDescriptors;
    MatOfKeyPoint[] multiImageKeypoints;

    private boolean inPrototypeMode = false;
    private static final String proto = "Proto", spatial = "Spatial";
    private static final String orb = "ORB", akaze = "AKAZE", brisk = "BRISK";

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    try {
                        initializeParams();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeParams() throws IOException {
        mOpenCvCameraView.enableView();

        modeButton = (Button) findViewById(R.id.modeButton);
        modeButton.setText(spatial);
        protoDisplayButton = (Button) findViewById(R.id.protoDisplayButton);
        protoDisplayButton.setText(orb);

        detectionORB = new long[statisticsLimit];
        detectionAKAZE = new long[statisticsLimit];
        detectionBRISK = new long[statisticsLimit];

        descriptionORB = new long[statisticsLimit];
        descriptionAKAZE = new long[statisticsLimit];
        descriptionBRISK = new long[statisticsLimit];

        matchingORB = new long[statisticsLimit];
        matchingAKAZE = new long[statisticsLimit];
        matchingBRISK = new long[statisticsLimit];

        detectorORB = ORB.create();
        detectorAKAZE = AKAZE.create();
        detectorBRISK = BRISK.create();

        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        AssetManager assetManager = getAssets();
        InputStream istr;
        Bitmap bitmap;

        //Initialize prototype for testing various methods and combinations
        protoImage = new Mat();

        istr = assetManager.open("usb_0.jpg");

        bitmap = BitmapFactory.decodeStream(istr);
        Utils.bitmapToMat(bitmap, protoImage);
        Imgproc.cvtColor(protoImage, protoImage, Imgproc.COLOR_RGB2GRAY);
        protoImage.convertTo(protoImage, 0);

        protoDescriptorsORB = new Mat();
        protoDescriptorsAKAZE = new Mat();
        protoDescriptorsBRISK = new Mat();

        protoKeypointsORB = new MatOfKeyPoint();
        protoKeypointsAKAZE = new MatOfKeyPoint();
        protoKeypointsBRISK = new MatOfKeyPoint();

        detectorORB.detect(protoImage, protoKeypointsORB);
        detectorORB.compute(protoImage, protoKeypointsORB, protoDescriptorsORB);

        detectorAKAZE.detect(protoImage, protoKeypointsAKAZE);
        detectorAKAZE.compute(protoImage, protoKeypointsAKAZE, protoDescriptorsAKAZE);

        detectorBRISK.detect(protoImage, protoKeypointsBRISK);
        detectorBRISK.compute(protoImage, protoKeypointsBRISK, protoDescriptorsBRISK);

        //Initialize spatial recognition array. Images may be changed at will.
        String[] files = assetManager.list("usb");

        List<String> fileNames = new LinkedList<String>(Arrays.asList(files));

        multiImageArray = new Mat[fileNames.size()];
        multiImageDescriptors = new Mat[fileNames.size()];
        multiImageKeypoints = new MatOfKeyPoint[fileNames.size()];
        int imageCount = 0;

        for (String fileName: fileNames
        ) {
            istr = assetManager.open("usb/" + fileName);

            bitmap = BitmapFactory.decodeStream(istr);
            multiImageArray[imageCount] = new Mat();
            Utils.bitmapToMat(bitmap, multiImageArray[imageCount]);

            Imgproc.cvtColor(multiImageArray[imageCount], multiImageArray[imageCount], Imgproc.COLOR_RGB2GRAY);
            multiImageArray[imageCount].convertTo(multiImageArray[imageCount], 0);

            multiImageKeypoints[imageCount] = new MatOfKeyPoint();
            multiImageDescriptors[imageCount] = new Mat();
            detectorORB.detect(multiImageArray[imageCount], multiImageKeypoints[imageCount]);
            detectorORB.compute(multiImageArray[imageCount], multiImageKeypoints[imageCount], multiImageDescriptors[imageCount]);

            imageCount++;
        }

        Log.d("Finished initialization", imageCount + " units in multi image array");
    }

    public MainActivity() {
    }

    public void switchModes(View v)
    {
        if (inPrototypeMode) {
            modeButton.setText(spatial);
            inPrototypeMode = false;
        } else {
            modeButton.setText(proto);
            inPrototypeMode = true;
        }
    }

    public void switchDisplay(View v) {
        switch (protoDisplayMode) {
            case 0:
                protoDisplayMode = 1;
                protoDisplayButton.setText(akaze);
                break;
            case 1:
                protoDisplayMode = 2;
                protoDisplayButton.setText(brisk);
                break;
            default:
                protoDisplayMode = 0;
                protoDisplayButton.setText(orb);
                break;
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_PERMISSION);
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.primary);
        mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        super.onPause();
    }

    @Override
    public void onResume() {
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        super.onResume();
    }

    public void onDestroy() {
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        super.onDestroy();
    }

    public void onCameraViewStarted(int cameraWidth, int cameraHeight) {
        Toast.makeText(this, "Setting dimensions: " + cameraWidth + "x" + cameraHeight,
                Toast.LENGTH_LONG).show();
        width = cameraWidth;
        height = cameraHeight;
    }

    public void onCameraViewStopped() {
    }

    public Mat protoMatch(Mat frame) {
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
        if (currentProtoFrame == statisticsLimit) {
            printStatistics();
            currentProtoFrame++;
            return frame;
        } else if (currentProtoFrame > statisticsLimit)
            return frame;

        MatOfKeyPoint frameKeypointsORB, frameKeypointsAKAZE, frameKeypointsBRISK;

        frameKeypointsORB = new MatOfKeyPoint();
        frameKeypointsAKAZE = new MatOfKeyPoint();
        frameKeypointsBRISK = new MatOfKeyPoint();

        Mat frameDescriptorsORB = new Mat();
        Mat frameDescriptorsAKAZE = new Mat();
        Mat frameDescriptorsBRISK = new Mat();

        long timestamp = System.nanoTime();
        detectorORB.detect(frame, frameKeypointsORB);
        detectionORB[currentProtoFrame] = System.nanoTime() - timestamp;

        timestamp = System.nanoTime();
        detectorAKAZE.detect(frame, frameKeypointsAKAZE);
        detectionAKAZE[currentProtoFrame] = System.nanoTime() - timestamp;

        timestamp = System.nanoTime();
        detectorBRISK.detect(frame, frameKeypointsBRISK);
        detectionBRISK[currentProtoFrame] = System.nanoTime() - timestamp;

        timestamp = System.nanoTime();
        detectorORB.compute(frame, frameKeypointsORB, frameDescriptorsORB);
        descriptionORB[currentProtoFrame] = System.nanoTime() - timestamp;

        timestamp = System.nanoTime();
        detectorAKAZE.compute(frame, frameKeypointsAKAZE, frameDescriptorsAKAZE);
        descriptionAKAZE[currentProtoFrame] = System.nanoTime() - timestamp;

        timestamp = System.nanoTime();
        detectorBRISK.compute(frame, frameKeypointsBRISK, frameDescriptorsBRISK);
        descriptionBRISK[currentProtoFrame] = System.nanoTime() - timestamp;

        MatOfDMatch matchesORB = new MatOfDMatch();
        MatOfDMatch matchesAKAZE = new MatOfDMatch();
        MatOfDMatch matchesBRISK = new MatOfDMatch();

        if (frame.type() == protoImage.type()) {
            timestamp = System.nanoTime();
            matcher.match(protoDescriptorsORB, frameDescriptorsORB, matchesORB);
            matchingORB[currentProtoFrame] = System.nanoTime() - timestamp;

            timestamp = System.nanoTime();
            matcher.match(protoDescriptorsAKAZE, frameDescriptorsAKAZE, matchesAKAZE);
            matchingAKAZE[currentProtoFrame] = System.nanoTime() - timestamp;

            timestamp = System.nanoTime();
            matcher.match(protoDescriptorsBRISK, frameDescriptorsBRISK, matchesBRISK);
            matchingBRISK[currentProtoFrame] = System.nanoTime() - timestamp;
        } else
            return frame;

        MatOfDMatch bestMatchesORB, bestMatchesAKAZE, bestMatchesBRISK;

        bestMatchesORB = new MatOfDMatch();
        bestMatchesAKAZE = new MatOfDMatch();
        bestMatchesBRISK = new MatOfDMatch();

        bestMatchesORB.fromList(getBestMatches(matchesORB.toList(), minDist));
        avgMatchesORB += bestMatchesORB.rows();
        bestMatchesAKAZE.fromList(getBestMatches(matchesAKAZE.toList(), minDist));
        avgMatchesAKAZE += bestMatchesAKAZE.rows();
        bestMatchesBRISK.fromList(getBestMatches(matchesBRISK.toList(), minDist));
        avgMatchesBRISK += bestMatchesBRISK.rows();

        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();

        if (frame.empty() || frame.cols() < 1 || frame.rows() < 1) {
            return frame;
        }
        switch (protoDisplayMode) {
            case 0:
                Features2d.drawMatches(protoImage, protoKeypointsORB, frame, frameKeypointsORB, bestMatchesORB, outputImg, GREEN, GREEN, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
                break;
            case 1:
                Features2d.drawMatches(protoImage, protoKeypointsAKAZE, frame, frameKeypointsAKAZE, bestMatchesAKAZE, outputImg, RED, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
                break;
            case 2:
                Features2d.drawMatches(protoImage, protoKeypointsBRISK, frame, frameKeypointsBRISK, bestMatchesBRISK, outputImg, BLUE, BLUE, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
                break;
        }
        Imgproc.resize(outputImg, outputImg, frame.size());
        currentProtoFrame++;
        return outputImg;
    }

    public Mat spatialMatch(Mat frame) {
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
        MatOfKeyPoint frameKeypoints = new MatOfKeyPoint();
        Mat frameDescriptors = new Mat();

        detectorORB.detect(frame, frameKeypoints);
        detectorORB.compute(frame, frameKeypoints, frameDescriptors);

        //Attempt to "perceive" in 3D by using multiple shifted images of the same object
        int bestMatchImage = 0, testCount = 0;
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch bestMatches = new MatOfDMatch();
        for (Mat multiImageDescriptor: multiImageDescriptors
             ) {
            if (multiImageArray[testCount].type() == frame.type())
                matcher.match(multiImageDescriptor, frameDescriptors, matches);
            else
                return frame;

            MatOfDMatch bestFrameMatches = new MatOfDMatch();
            bestFrameMatches.fromList(getBestMatches(matches.toList(), minDist));

            if (bestFrameMatches.height() * bestFrameMatches.width() >
                    bestMatches.height() * bestMatches.width()) {
                bestMatches = bestFrameMatches;
                bestMatchImage = testCount;
            }

            testCount++;
        }

        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();

        if (frame.empty() || frame.cols() < 1 || frame.rows() < 1) {
            return frame;
        }
        Features2d.drawMatches(multiImageArray[bestMatchImage], multiImageKeypoints[bestMatchImage], frame, frameKeypoints, bestMatches, outputImg, RED, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
        Imgproc.resize(outputImg, outputImg, frame.size());

        return outputImg;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (inPrototypeMode)
            return protoMatch(inputFrame.rgba());
        else
            return spatialMatch(inputFrame.rgba());
    }

    public LinkedList<DMatch> getBestMatches(List<DMatch> matches, double threshold) {
        LinkedList<DMatch> bestMatches = new LinkedList<DMatch>();
        for(int i = 0;i < matches.size(); i++){
            if (matches.get(i).distance <= threshold)
                bestMatches.addLast(matches.get(i));
        }
        return bestMatches;
    }

    public void printStatistics() {
        DecimalFormat df = new DecimalFormat("0.00");
        String _detectionORB = new String(), _detectionAKAZE = new String(), _detectionBRISK = new String(),
                _descriptionORB = new String(), _descriptionAKAZE = new String(), _descriptionBRISK = new String(),
                _matchingORB = new String(), _matchingAKAZE = new String(), _matchingBRISK = new String();

        for (int i = 0; i < statisticsLimit; i++) {
            _detectionORB = _detectionORB + df.format((double) detectionORB[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _detectionAKAZE = _detectionAKAZE + df.format((double) detectionAKAZE[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _detectionBRISK = _detectionBRISK + df.format((double) detectionBRISK[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");

            _descriptionORB = _descriptionORB + df.format((double) descriptionORB[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _descriptionAKAZE = _descriptionAKAZE + df.format((double) descriptionAKAZE[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _descriptionBRISK = _descriptionBRISK + df.format((double) descriptionBRISK[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");

            _matchingORB = _matchingORB + df.format((double) matchingORB[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _matchingAKAZE = _matchingAKAZE + df.format((double) matchingAKAZE[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
            _matchingBRISK = _matchingBRISK + df.format((double) matchingBRISK[i] / 1000000) + (i == statisticsLimit - 1 ? "," : "");
        }

        System.out.println("ORB detection statistics: " + _detectionORB);
        System.out.println("ORB description statistics: " + _descriptionORB);
        System.out.println("ORB matching statistics: " + _matchingORB);
        System.out.println("ORB keypoint statistics: " + avgMatchesORB / statisticsLimit);

        System.out.println("AKAZE detection statistics: " + _detectionAKAZE);
        System.out.println("AKAZE description statistics: " + _descriptionAKAZE);
        System.out.println("AKAZE matching statistics: " + _matchingAKAZE);
        System.out.println("AKAZE keypoint statistics: " + avgMatchesAKAZE / statisticsLimit);

        System.out.println("BRISK detection statistics: " + _detectionBRISK);
        System.out.println("BRISK description statistics: " + _descriptionBRISK);
        System.out.println("BRISK matching statistics: " + _matchingBRISK);
        System.out.println("BRISK keypoint statistics: " + avgMatchesBRISK / statisticsLimit);

    }
}
