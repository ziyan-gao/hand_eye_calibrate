#pragma once

#include <array>
#include <memory>
#include <string>

#include <QLabel>
#include <QTimer>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/panel.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "camera_extrinsic_calibration_interfaces/action/solve.hpp"
#include "camera_extrinsic_calibration_interfaces/msg/calibration_status.hpp"
#include "camera_extrinsic_calibration_interfaces/srv/capture.hpp"
#include "camera_extrinsic_calibration_interfaces/srv/configure.hpp"
#include "camera_extrinsic_calibration_interfaces/srv/delete_sample.hpp"
#include "camera_extrinsic_calibration_interfaces/srv/load_session.hpp"
#include "camera_extrinsic_calibration_interfaces/srv/save_result.hpp"

class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QCheckBox;

namespace camera_extrinsic_calibration_rviz
{

namespace cec = camera_extrinsic_calibration_interfaces;

class CalibrationPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit CalibrationPanel(QWidget * parent = nullptr);
  ~CalibrationPanel() override;

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private Q_SLOTS:
  void onApplyConfig();
  void onCapture();
  void onDeleteLast();
  void onSolve();
  void onSave();
  void onLoadSession();
  void onWatchdog();
  void onToggleManual();

private:
  using PoseRow = std::array<QLabel *, 6>;
  using SolveGoalHandle = rclcpp_action::ClientGoalHandle<cec::action::Solve>;

  void buildUi();
  void handleStatus(cec::msg::CalibrationStatus::SharedPtr msg);
  void handleImage(sensor_msgs::msg::Image::SharedPtr msg);
  void setPoseRow(PoseRow & row, const double v[6]);
  void setPoseRowNull(PoseRow & row);
  void updateButtons();
  void setMessage(const QString & text, bool error = false);
  bool serviceReady(rclcpp::ClientBase & client);
  // Post a callable from a ROS callback onto the Qt GUI thread.
  template<class F>
  void postToGui(F && f);

  // ROS
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<cec::msg::CalibrationStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Client<cec::srv::Configure>::SharedPtr configure_client_;
  rclcpp::Client<cec::srv::Capture>::SharedPtr capture_client_;
  rclcpp::Client<cec::srv::DeleteSample>::SharedPtr delete_client_;
  rclcpp::Client<cec::srv::SaveResult>::SharedPtr save_client_;
  rclcpp::Client<cec::srv::LoadSession>::SharedPtr load_client_;
  rclcpp_action::Client<cec::action::Solve>::SharedPtr solve_client_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr manual_client_;
  SolveGoalHandle::SharedPtr solve_goal_;
  std::shared_ptr<bool> alive_;

  // state
  cec::msg::CalibrationStatus status_;
  bool have_status_ = false;
  rclcpp::Time last_status_time_{0, 0, RCL_STEADY_TIME};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};
  bool pending_ = false;

  // widgets
  QSpinBox * columns_ = nullptr;
  QSpinBox * rows_ = nullptr;
  QDoubleSpinBox * square_ = nullptr;
  QComboBox * mode_ = nullptr;
  QLineEdit * session_name_ = nullptr;
  QPushButton * apply_btn_ = nullptr;
  QLabel * config_state_ = nullptr;
  QLabel * image_label_ = nullptr;
  QLabel * detect_state_ = nullptr;
  PoseRow eef_row_{};
  PoseRow cam_row_{};
  QLabel * eef_reason_ = nullptr;
  QLabel * cam_reason_ = nullptr;
  QLabel * samples_label_ = nullptr;
  QLabel * robot_mode_label_ = nullptr;
  QPushButton * manual_btn_ = nullptr;
  QPushButton * capture_btn_ = nullptr;
  QPushButton * delete_btn_ = nullptr;
  QPushButton * solve_btn_ = nullptr;
  QComboBox * method_ = nullptr;
  QPushButton * save_btn_ = nullptr;
  QLineEdit * result_name_ = nullptr;
  QCheckBox * overwrite_ = nullptr;
  QLineEdit * load_path_ = nullptr;
  QPushButton * load_btn_ = nullptr;
  QPlainTextEdit * result_text_ = nullptr;
  QLabel * message_ = nullptr;
  QTimer * watchdog_ = nullptr;
};

}  // namespace camera_extrinsic_calibration_rviz
