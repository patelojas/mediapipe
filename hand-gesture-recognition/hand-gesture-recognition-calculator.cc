#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe
{

  enum class FingerState
  {
    UNKNOW,
    OPEN,
    CLOSE
  };

  namespace
  {
    constexpr char normRectTag[] = "NORM_RECT";
    constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
    constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
  } // namespace

  // Graph config:
  //
  // node {
  //   calculator: "HandGestureRecognitionCalculator"
  //   input_stream: "NORM_LANDMARKS:scaled_landmarks"
  //   input_stream: "NORM_RECT:hand_rect_for_next_frame"
  // }
  class HandGestureRecognitionCalculator : public CalculatorBase
  {
  public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc);
    ::mediapipe::Status Open(CalculatorContext *cc) override;

    ::mediapipe::Status Process(CalculatorContext *cc) override;

  private:
    float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
    {
      float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
      return std::sqrt(dist);
    }

    bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
    {
      float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
      return distance < 0.1;
    }
  };

  REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

  ::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
      CalculatorContract *cc)
  {
    RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

    RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
    cc->Outputs().Tag(recognizedHandGestureTag).Set<std::string>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status HandGestureRecognitionCalculator::Open(
      CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status HandGestureRecognitionCalculator::Process(
      CalculatorContext *cc)
  {
    std::string *recognized_hand_gesture;

    // hand closed (red) rectangle
    const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
    float width = rect->width();
    float height = rect->height();

    if (width < 0.01 || height < 0.01)
    {
      // LOG(INFO) << "No Hand Detected";
      recognized_hand_gesture = new std::string("___");
      cc->Outputs()
          .Tag(recognizedHandGestureTag)
          .Add(recognized_hand_gesture, cc->InputTimestamp());
      return ::mediapipe::OkStatus();
    }

    const auto &landmarkList = cc->Inputs()
                                   .Tag(normalizedLandmarkListTag)
                                   .Get<mediapipe::NormalizedLandmarkList>();
    RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

    // finger states
    FingerState thumbState = FingerState::UNKNOW;
    FingerState firstFingerState = FingerState::UNKNOW;
    FingerState secondFingerState = FingerState::UNKNOW;
    FingerState thirdFingerState = FingerState::UNKNOW;
    FingerState fourthFingerState = FingerState::UNKNOW;
    //
    float threshold = 0.01;
    float closeFingerThreshold = 0.01;

    float pseudoFixKeyPoint = landmarkList.landmark(2).x();
    if (landmarkList.landmark(3).x() + threshold < pseudoFixKeyPoint && landmarkList.landmark(4).x() + threshold < landmarkList.landmark(3).x())
    {
      thumbState = FingerState::OPEN;
    }
    else if (pseudoFixKeyPoint + closeFingerThreshold < landmarkList.landmark(3).x() && landmarkList.landmark(3).x() + closeFingerThreshold < landmarkList.landmark(4).x())
    {
      thumbState = FingerState::CLOSE;
    }

    pseudoFixKeyPoint = landmarkList.landmark(6).y();
    if (landmarkList.landmark(7).y() + threshold < pseudoFixKeyPoint && landmarkList.landmark(8).y() + threshold < landmarkList.landmark(7).y())
    {
      firstFingerState = FingerState::OPEN;
    }
    else if (pseudoFixKeyPoint + closeFingerThreshold < landmarkList.landmark(7).y() && landmarkList.landmark(7).y() + closeFingerThreshold < landmarkList.landmark(8).y())
    {
      firstFingerState = FingerState::CLOSE;
    }

    pseudoFixKeyPoint = landmarkList.landmark(10).y();
    if (landmarkList.landmark(11).y() + threshold < pseudoFixKeyPoint && landmarkList.landmark(12).y() + threshold < landmarkList.landmark(11).y())
    {
      secondFingerState = FingerState::OPEN;
    }
    else if (pseudoFixKeyPoint + closeFingerThreshold < landmarkList.landmark(11).y() && landmarkList.landmark(11).y() + closeFingerThreshold < landmarkList.landmark(12).y())
    {
      secondFingerState = FingerState::CLOSE;
    }

    pseudoFixKeyPoint = landmarkList.landmark(14).y();
    if (landmarkList.landmark(15).y() + threshold < pseudoFixKeyPoint && landmarkList.landmark(16).y() + threshold < landmarkList.landmark(15).y())
    {
      thirdFingerState = FingerState::OPEN;
    }
    else if (pseudoFixKeyPoint + closeFingerThreshold < landmarkList.landmark(15).y() && landmarkList.landmark(15).y() + closeFingerThreshold < landmarkList.landmark(16).y())
    {
      thirdFingerState = FingerState::CLOSE;
    }
    pseudoFixKeyPoint = landmarkList.landmark(18).y();
    if (landmarkList.landmark(19).y() + threshold < pseudoFixKeyPoint && landmarkList.landmark(20).y() + threshold < landmarkList.landmark(19).y())
    {
      fourthFingerState = FingerState::OPEN;
    }
    else if (pseudoFixKeyPoint + closeFingerThreshold < landmarkList.landmark(19).y() && landmarkList.landmark(19).y() + closeFingerThreshold < landmarkList.landmark(20).y())
    {
      fourthFingerState = FingerState::CLOSE;
    }

    // Hand gesture recognition
    if (thumbState == FingerState::OPEN && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::OPEN && thirdFingerState == FingerState::OPEN && fourthFingerState == FingerState::OPEN)
    {
      recognized_hand_gesture = new std::string("FIVE");
    }
    else if (thumbState == FingerState::CLOSE && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::OPEN && thirdFingerState == FingerState::OPEN && fourthFingerState == FingerState::OPEN)
    {
      recognized_hand_gesture = new std::string("FOUR");
    }
    else if (thumbState == FingerState::OPEN && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::OPEN && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::CLOSE)
    {
      recognized_hand_gesture = new std::string("TREE");
    }
    else if (thumbState == FingerState::OPEN && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::CLOSE && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::CLOSE)
    {
      recognized_hand_gesture = new std::string("TWO");
    }
    else if (thumbState == FingerState::CLOSE && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::CLOSE && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::CLOSE)
    {
      recognized_hand_gesture = new std::string("ONE");
    }
    else if (thumbState == FingerState::CLOSE && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::OPEN && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::CLOSE)
    {
      recognized_hand_gesture = new std::string("YEAH");
    }
    else if (thumbState == FingerState::CLOSE && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::CLOSE && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::OPEN)
    {
      recognized_hand_gesture = new std::string("ROCK");
    }
    else if (thumbState == FingerState::OPEN && firstFingerState == FingerState::OPEN && secondFingerState == FingerState::CLOSE && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::OPEN)
    {
      recognized_hand_gesture = new std::string("SPIDERMAN");
    }
    else if (thumbState == FingerState::CLOSE && firstFingerState == FingerState::CLOSE && secondFingerState == FingerState::CLOSE && thirdFingerState == FingerState::CLOSE && fourthFingerState == FingerState::CLOSE)
    {
      recognized_hand_gesture = new std::string("FIST");
    }
    else if (firstFingerState == FingerState::CLOSE && secondFingerState == FingerState::OPEN && thirdFingerState == FingerState::OPEN && fourthFingerState == FingerState::OPEN && this->isThumbNearFirstFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    {
      recognized_hand_gesture = new std::string("OK");
    }
    else
    {
      recognized_hand_gesture = new std::string("___");
      // LOG(INFO) << "Finger States: " << thumbState == FingerState::OPEN << firstFingerState == FingerState::OPEN << secondFingerState == FingerState::OPEN << thirdFingerState == FingerState::OPEN << fourthFingerState == FingerState::OPEN;
    }
    // LOG(INFO) << recognized_hand_gesture;

    cc->Outputs()
        .Tag(recognizedHandGestureTag)
        .Add(recognized_hand_gesture, cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  } // namespace mediapipe

} // namespace mediapipe
