#include "camera_extrinsic_calibration_rviz/calibration_panel.hpp"

#include <cmath>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFontDatabase>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLineEdit>
#include <QMetaObject>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace camera_extrinsic_calibration_rviz
{

namespace
{
using Status = cec::msg::CalibrationStatus;

// Fixed-width numeric field: sign + 5 integer digits + '.' + 4 decimals = 11 chars.
constexpr int kFieldChars = 11;
constexpr double kFieldMax = 99999.99995;

QString fieldText(double v)
{
  if (!std::isfinite(v) || std::fabs(v) >= kFieldMax) {
    return QString("invalid").rightJustified(kFieldChars, ' ');
  }
  return QString::asprintf("%+011.4f", v);
}

QString nullText() {return QString("null").rightJustified(kFieldChars, ' ');}

struct Mat3 {double m[3][3];};

Mat3 quatToMat(double x, double y, double z, double w)
{
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n; y /= n; z /= n; w /= n;
  return {{
    {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}

// R = Rz(yaw) Ry(pitch) Rx(roll); returns true if near gimbal lock.
bool matToRpyDeg(const Mat3 & R, double out[3])
{
  const double sp = std::max(-1.0, std::min(1.0, -R.m[2][0]));
  const double pitch = std::asin(sp);
  const bool singular = std::fabs(sp) > std::cos(M_PI / 180.0);
  double roll, yaw;
  if (singular) {
    roll = 0.0;
    yaw = std::atan2(-R.m[0][1], R.m[1][1]);
  } else {
    roll = std::atan2(R.m[2][1], R.m[2][2]);
    yaw = std::atan2(R.m[1][0], R.m[0][0]);
  }
  out[0] = roll * 180.0 / M_PI;
  out[1] = pitch * 180.0 / M_PI;
  out[2] = yaw * 180.0 / M_PI;
  return singular;
}

// Transform -> [x,y,z mm, roll,pitch,yaw deg]; invert=true gives the inverse transform.
bool transformToDisplay(const geometry_msgs::msg::Transform & t, bool invert, double v[6])
{
  Mat3 R = quatToMat(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w);
  double p[3] = {t.translation.x, t.translation.y, t.translation.z};
  if (invert) {
    Mat3 Rt;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {Rt.m[i][j] = R.m[j][i];}
    }
    double q[3];
    for (int i = 0; i < 3; ++i) {
      q[i] = -(Rt.m[i][0] * p[0] + Rt.m[i][1] * p[1] + Rt.m[i][2] * p[2]);
    }
    R = Rt;
    p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
  }
  for (int i = 0; i < 3; ++i) {v[i] = p[i] * 1000.0;}
  return matToRpyDeg(R, v + 3);
}

QString stateName(uint8_t s)
{
  switch (s) {
    case Status::UNCONFIGURED: return "UNCONFIGURED";
    case Status::WAITING_FOR_DATA: return "WAITING_FOR_DATA";
    case Status::READY: return "READY";
    case Status::CAPTURING: return "CAPTURING";
    case Status::SOLVING: return "SOLVING";
    default: return "ERROR";
  }
}

QString resultStateName(uint8_t s)
{
  switch (s) {
    case Status::RESULT_VALID: return "VALID";
    case Status::RESULT_STALE: return "STALE (dataset changed, solve again)";
    case Status::RESULT_FAILED: return "FAILED";
    default: return "NONE";
  }
}
}  // namespace

CalibrationPanel::CalibrationPanel(QWidget * parent)
: rviz_common::Panel(parent), alive_(std::make_shared<bool>(true))
{
  buildUi();
}

CalibrationPanel::~CalibrationPanel()
{
  *alive_ = false;
  if (solve_goal_ && solve_client_) {
    solve_client_->async_cancel_goal(solve_goal_);
  }
  status_sub_.reset();
  image_sub_.reset();
}

template<class F>
void CalibrationPanel::postToGui(F && f)
{
  QMetaObject::invokeMethod(this, std::forward<F>(f), Qt::QueuedConnection);
}

void CalibrationPanel::buildUi()
{
  const QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  auto * root = new QVBoxLayout(this);

  // ---- configuration
  auto * cfg_box = new QGroupBox("Configuration");
  auto * cfg = new QGridLayout(cfg_box);
  columns_ = new QSpinBox; columns_->setRange(3, 50); columns_->setValue(9);
  rows_ = new QSpinBox; rows_->setRange(3, 50); rows_->setValue(6);
  square_ = new QDoubleSpinBox; square_->setRange(0.1, 500.0); square_->setDecimals(2);
  square_->setValue(26.5); square_->setSuffix(" mm");
  mode_ = new QComboBox; mode_->addItem("Eye in hand", "eye_in_hand");
  mode_->addItem("Eye out of hand", "eye_to_hand");
  session_name_ = new QLineEdit; session_name_->setPlaceholderText("session name (optional)");
  apply_btn_ = new QPushButton("Apply Config");
  config_state_ = new QLabel("not configured");
  cfg->addWidget(new QLabel("Inner corners (cols x rows)"), 0, 0);
  auto * cr = new QHBoxLayout; cr->addWidget(columns_); cr->addWidget(new QLabel("x"));
  cr->addWidget(rows_);
  cfg->addLayout(cr, 0, 1);
  cfg->addWidget(new QLabel("Square size"), 1, 0); cfg->addWidget(square_, 1, 1);
  cfg->addWidget(new QLabel("Mode"), 2, 0); cfg->addWidget(mode_, 2, 1);
  cfg->addWidget(new QLabel("Session"), 3, 0); cfg->addWidget(session_name_, 3, 1);
  cfg->addWidget(apply_btn_, 4, 0); cfg->addWidget(config_state_, 4, 1);
  root->addWidget(cfg_box);

  // ---- image
  image_label_ = new QLabel("no preview");
  image_label_->setAlignment(Qt::AlignCenter);
  image_label_->setMinimumSize(320, 180);
  image_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  image_label_->setStyleSheet("background-color: #202020; color: #aaaaaa;");
  root->addWidget(image_label_, 1);
  detect_state_ = new QLabel("board: -");
  root->addWidget(detect_state_);

  // ---- live poses (fixed geometry)
  auto * pose_box = new QGroupBox("Live poses (mm, deg; RPY = fixed-axis XYZ)");
  auto * pg = new QGridLayout(pose_box);
  const char * headers[6] = {"X", "Y", "Z", "Roll", "Pitch", "Yaw"};
  QFontMetrics fm(mono);
  const int field_w = fm.horizontalAdvance(QString(kFieldChars, '0')) + 8;
  for (int i = 0; i < 6; ++i) {
    auto * h = new QLabel(headers[i]);
    h->setAlignment(Qt::AlignRight);
    h->setFixedWidth(field_w);
    pg->addWidget(h, 0, i + 1);
  }
  auto make_row = [&](PoseRow & row, const QString & name, int r) {
      pg->addWidget(new QLabel(name), r, 0);
      for (int i = 0; i < 6; ++i) {
        auto * l = new QLabel(nullText());
        l->setFont(mono);
        l->setFixedWidth(field_w);
        l->setFixedHeight(fm.height() + 4);
        l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        row[i] = l;
        pg->addWidget(l, r, i + 1);
      }
    };
  make_row(eef_row_, "EEF in base", 1);
  make_row(cam_row_, "Camera in board", 3);
  eef_reason_ = new QLabel; cam_reason_ = new QLabel;
  for (auto * l : {eef_reason_, cam_reason_}) {
    l->setStyleSheet("color: #c08000;");
    l->setFixedHeight(fm.height() + 2);
    l->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
  }
  pg->addWidget(eef_reason_, 2, 1, 1, 6);
  pg->addWidget(cam_reason_, 4, 1, 1, 6);
  root->addWidget(pose_box);

  // ---- actions
  auto * act_box = new QGroupBox("Samples & solve");
  auto * ag = new QGridLayout(act_box);
  samples_label_ = new QLabel("Valid samples: 0");
  capture_btn_ = new QPushButton("Capture");
  delete_btn_ = new QPushButton("Delete Last");
  method_ = new QComboBox;
  for (auto m : {"PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS"}) {method_->addItem(m);}
  solve_btn_ = new QPushButton("Solve");
  result_name_ = new QLineEdit; result_name_->setPlaceholderText("result name (optional)");
  overwrite_ = new QCheckBox("overwrite");
  save_btn_ = new QPushButton("Save Extrinsics");
  load_path_ = new QLineEdit; load_path_->setPlaceholderText("session dir to load");
  load_btn_ = new QPushButton("Load Session");
  robot_mode_label_ = new QLabel("Robot mode: -");
  manual_btn_ = new QPushButton("Manual Mode");
  ag->addWidget(robot_mode_label_, 5, 0, 1, 2); ag->addWidget(manual_btn_, 5, 2);
  ag->addWidget(samples_label_, 0, 0, 1, 3);
  ag->addWidget(capture_btn_, 1, 0); ag->addWidget(delete_btn_, 1, 1);
  ag->addWidget(method_, 2, 0); ag->addWidget(solve_btn_, 2, 1);
  ag->addWidget(result_name_, 3, 0); ag->addWidget(overwrite_, 3, 1);
  ag->addWidget(save_btn_, 3, 2);
  ag->addWidget(load_path_, 4, 0, 1, 2); ag->addWidget(load_btn_, 4, 2);
  root->addWidget(act_box);

  // ---- result
  result_text_ = new QPlainTextEdit;
  result_text_->setReadOnly(true);
  result_text_->setFont(mono);
  result_text_->setFixedHeight(fm.height() * 9);
  root->addWidget(result_text_);
  message_ = new QLabel("waiting for backend");
  message_->setWordWrap(true);
  root->addWidget(message_);

  connect(apply_btn_, &QPushButton::clicked, this, &CalibrationPanel::onApplyConfig);
  connect(capture_btn_, &QPushButton::clicked, this, &CalibrationPanel::onCapture);
  connect(delete_btn_, &QPushButton::clicked, this, &CalibrationPanel::onDeleteLast);
  connect(solve_btn_, &QPushButton::clicked, this, &CalibrationPanel::onSolve);
  connect(save_btn_, &QPushButton::clicked, this, &CalibrationPanel::onSave);
  connect(load_btn_, &QPushButton::clicked, this, &CalibrationPanel::onLoadSession);
  connect(manual_btn_, &QPushButton::clicked, this, &CalibrationPanel::onToggleManual);
  updateButtons();
}

void CalibrationPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  std::weak_ptr<bool> alive = alive_;
  status_sub_ = node_->create_subscription<Status>(
    "/calibration/status", 10, [this, alive](Status::SharedPtr msg) {
      if (auto a = alive.lock(); a && *a) {postToGui([this, msg] {handleStatus(msg);});}
    });
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
    "/calibration/preview", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
    [this, alive](sensor_msgs::msg::Image::SharedPtr msg) {
      if (auto a = alive.lock(); a && *a) {postToGui([this, msg] {handleImage(msg);});}
    });
  configure_client_ = node_->create_client<cec::srv::Configure>("/calibration/configure");
  capture_client_ = node_->create_client<cec::srv::Capture>("/calibration/capture");
  delete_client_ = node_->create_client<cec::srv::DeleteSample>("/calibration/delete_sample");
  save_client_ = node_->create_client<cec::srv::SaveResult>("/calibration/save_result");
  load_client_ = node_->create_client<cec::srv::LoadSession>("/calibration/load_session");
  manual_client_ = node_->create_client<std_srvs::srv::SetBool>("/calibration/set_manual_mode");
  solve_client_ = rclcpp_action::create_client<cec::action::Solve>(node_, "/calibration/solve");

  watchdog_ = new QTimer(this);
  connect(watchdog_, &QTimer::timeout, this, &CalibrationPanel::onWatchdog);
  watchdog_->start(250);
}

// ------------------------------------------------------------------ incoming data
void CalibrationPanel::handleStatus(Status::SharedPtr msg)
{
  status_ = *msg;
  have_status_ = true;
  last_status_time_ = steady_clock_.now();

  if (msg->session_id.empty()) {
    config_state_->setText("not configured");
  } else {
    config_state_->setText(QString("%1  rev %2  [%3x%4, %5 mm, %6]")
      .arg(QString::fromStdString(msg->session_id)).arg(msg->config_revision)
      .arg(msg->columns).arg(msg->rows).arg(msg->square_size_mm, 0, 'f', 2)
      .arg(QString::fromStdString(msg->mode)));
  }

  double v[6];
  if (msg->eef_valid) {
    transformToDisplay(msg->eef, false, v);
    setPoseRow(eef_row_, v);
  } else {
    setPoseRowNull(eef_row_);
  }
  eef_reason_->setText(QString::fromStdString(msg->eef_reason));

  if (msg->board_valid) {
    const bool singular = transformToDisplay(msg->board, true, v);  // board <- camera
    setPoseRow(cam_row_, v);
    QString reason = QString::fromStdString(msg->board_reason);
    if (singular) {reason = "RPY near singular (pitch ~ +/-90 deg). " + reason;}
    cam_reason_->setText(reason);
    detect_state_->setText("board: detected");
  } else {
    setPoseRowNull(cam_row_);
    cam_reason_->setText(QString::fromStdString(msg->board_reason));
    detect_state_->setText("board: not detected");
  }

  samples_label_->setText(QString("Valid samples: %1   (total %2, dataset rev %3)   state: %4")
    .arg(msg->num_valid_samples).arg(msg->num_total_samples).arg(msg->dataset_revision)
    .arg(stateName(msg->state)));

  if (msg->result_state == Status::RESULT_NONE) {
    result_text_->setPlainText("No result.");
  } else if (msg->result_state == Status::RESULT_FAILED) {
    result_text_->setPlainText("Last solve FAILED. See message below.");
  } else {
    const auto & r = msg->result;
    transformToDisplay(r.transform, false, v);
    const auto & q = r.transform.rotation;
    result_text_->setPlainText(
      QString("State : %1\nT_%2_from_%3 (%4, %5)\nxyz mm: %6 %7 %8\nrpy deg: %9 %10 %11\n")
      .arg(resultStateName(msg->result_state))
      .arg(QString::fromStdString(r.parent_frame)).arg(QString::fromStdString(r.child_frame))
      .arg(QString::fromStdString(r.mode)).arg(QString::fromStdString(r.method))
      .arg(fieldText(v[0])).arg(fieldText(v[1])).arg(fieldText(v[2]))
      .arg(fieldText(v[3])).arg(fieldText(v[4])).arg(fieldText(v[5])) +
      QString("quat xyzw: %1 %2 %3 %4\ndataset rev %5, %6 samples\n%7")
      .arg(q.x, 0, 'f', 6).arg(q.y, 0, 'f', 6).arg(q.z, 0, 'f', 6).arg(q.w, 0, 'f', 6)
      .arg(r.dataset_revision).arg(r.sample_ids.size())
      .arg(QString::fromStdString(r.report)));
  }
  const QString mode_name = msg->robot_mode == 2 ? "MANUAL (drag enabled)" :
    msg->robot_mode == 0 ? "POSITION (holding)" :
    msg->robot_mode < 0 ? "unknown" : QString("mode %1").arg(msg->robot_mode);
  robot_mode_label_->setText(QString("Robot mode: %1   state %2").arg(mode_name)
    .arg(msg->robot_state));
  manual_btn_->setText(msg->robot_mode == 2 ? "Lock Pose (position mode)" : "Manual Mode");
  if (!pending_) {setMessage(QString::fromStdString(msg->message));}
  updateButtons();
}

void CalibrationPanel::handleImage(sensor_msgs::msg::Image::SharedPtr msg)
{
  QImage::Format fmt;
  if (msg->encoding == "bgr8") {
    fmt = QImage::Format_BGR888;
  } else if (msg->encoding == "rgb8") {
    fmt = QImage::Format_RGB888;
  } else if (msg->encoding == "mono8") {
    fmt = QImage::Format_Grayscale8;
  } else {
    image_label_->setText(QString("unsupported encoding %1")
      .arg(QString::fromStdString(msg->encoding)));
    return;
  }
  QImage img(msg->data.data(), static_cast<int>(msg->width), static_cast<int>(msg->height),
    static_cast<int>(msg->step), fmt);
  image_label_->setPixmap(QPixmap::fromImage(img.copy()).scaled(
      image_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void CalibrationPanel::onWatchdog()
{
  const bool stale = !have_status_ ||
    (steady_clock_.now() - last_status_time_).seconds() > 1.0;
  if (stale) {
    have_status_ = false;
    setPoseRowNull(eef_row_);
    setPoseRowNull(cam_row_);
    eef_reason_->setText("backend not responding");
    cam_reason_->setText("backend not responding");
    detect_state_->setText("board: stale");
    updateButtons();
  }
}

// ------------------------------------------------------------------ display helpers
void CalibrationPanel::setPoseRow(PoseRow & row, const double v[6])
{
  for (int i = 0; i < 6; ++i) {row[i]->setText(fieldText(v[i]));}
}

void CalibrationPanel::setPoseRowNull(PoseRow & row)
{
  for (auto * l : row) {l->setText(nullText());}
}

void CalibrationPanel::setMessage(const QString & text, bool error)
{
  message_->setStyleSheet(error ? "color: #d04040;" : "");
  message_->setText(text);
}

void CalibrationPanel::updateButtons()
{
  const bool ok = have_status_ && !pending_ && !status_.busy;
  apply_btn_->setEnabled(have_status_ && !pending_ && !status_.busy);
  capture_btn_->setEnabled(ok && status_.can_capture);
  delete_btn_->setEnabled(ok && status_.num_valid_samples > 0);
  solve_btn_->setEnabled(ok && status_.can_solve);
  save_btn_->setEnabled(ok && status_.can_save);
  load_btn_->setEnabled(ok);
  manual_btn_->setEnabled(have_status_ && !pending_ && status_.robot_mode >= 0);
}

bool CalibrationPanel::serviceReady(rclcpp::ClientBase & client)
{
  if (!client.service_is_ready()) {
    setMessage(QString("service %1 not available").arg(client.get_service_name()), true);
    return false;
  }
  return true;
}

// ------------------------------------------------------------------ actions
void CalibrationPanel::onApplyConfig()
{
  if (!serviceReady(*configure_client_)) {return;}
  auto req = std::make_shared<cec::srv::Configure::Request>();
  req->columns = columns_->value();
  req->rows = rows_->value();
  req->square_size_mm = square_->value();
  req->mode = mode_->currentData().toString().toStdString();
  req->session_name = session_name_->text().trimmed().toStdString();
  pending_ = true;
  updateButtons();
  setMessage("configuring...");
  std::weak_ptr<bool> alive = alive_;
  configure_client_->async_send_request(req,
    [this, alive](rclcpp::Client<cec::srv::Configure>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            if (res->success) {session_name_->clear();}
            updateButtons();
          });
      }
    });
}

void CalibrationPanel::onCapture()
{
  if (!serviceReady(*capture_client_)) {return;}
  auto req = std::make_shared<cec::srv::Capture::Request>();
  req->request_id = std::to_string(steady_clock_.now().nanoseconds());
  req->expected_config_revision = status_.config_revision;
  pending_ = true;
  updateButtons();
  setMessage("capturing... keep the robot still");
  std::weak_ptr<bool> alive = alive_;
  capture_client_->async_send_request(req,
    [this, alive](rclcpp::Client<cec::srv::Capture>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            updateButtons();
          });
      }
    });
}

void CalibrationPanel::onDeleteLast()
{
  if (!serviceReady(*delete_client_)) {return;}
  auto req = std::make_shared<cec::srv::DeleteSample::Request>();
  req->sample_id = "last";
  pending_ = true;
  updateButtons();
  std::weak_ptr<bool> alive = alive_;
  delete_client_->async_send_request(req,
    [this, alive](rclcpp::Client<cec::srv::DeleteSample>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            updateButtons();
          });
      }
    });
}

void CalibrationPanel::onSolve()
{
  if (!solve_client_->action_server_is_ready()) {
    setMessage("solve action server not available", true);
    return;
  }
  cec::action::Solve::Goal goal;
  goal.method = method_->currentText().toStdString();
  goal.expected_dataset_revision = status_.dataset_revision;
  pending_ = true;
  updateButtons();
  setMessage("solving...");
  std::weak_ptr<bool> alive = alive_;
  auto opts = rclcpp_action::Client<cec::action::Solve>::SendGoalOptions();
  opts.goal_response_callback = [this, alive](SolveGoalHandle::SharedPtr gh) {
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, gh] {
            solve_goal_ = gh;
            if (!gh) {
              pending_ = false;
              setMessage("solve rejected by backend (busy or not configured)", true);
              updateButtons();
            }
          });
      }
    };
  opts.feedback_callback = [this, alive](SolveGoalHandle::SharedPtr,
      const std::shared_ptr<const cec::action::Solve::Feedback> fb) {
      if (auto a = alive.lock(); a && *a) {
        const QString text = QString("solving: %1 (%2%)")
          .arg(QString::fromStdString(fb->stage)).arg(static_cast<int>(fb->progress * 100));
        postToGui([this, text] {setMessage(text);});
      }
    };
  opts.result_callback = [this, alive](const SolveGoalHandle::WrappedResult & wr) {
      if (auto a = alive.lock(); a && *a) {
        const bool ok = wr.code == rclcpp_action::ResultCode::SUCCEEDED && wr.result &&
          wr.result->success;
        const QString text = wr.result ? QString::fromStdString(wr.result->message) :
          QString("solve ended without result");
        postToGui([this, ok, text] {
            pending_ = false;
            solve_goal_.reset();
            setMessage(text, !ok);
            updateButtons();
          });
      }
    };
  solve_client_->async_send_goal(goal, opts);
}

void CalibrationPanel::onSave()
{
  if (!serviceReady(*save_client_)) {return;}
  auto req = std::make_shared<cec::srv::SaveResult::Request>();
  req->name = result_name_->text().trimmed().toStdString();
  req->overwrite = overwrite_->isChecked();
  pending_ = true;
  updateButtons();
  std::weak_ptr<bool> alive = alive_;
  save_client_->async_send_request(req,
    [this, alive](rclcpp::Client<cec::srv::SaveResult>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            updateButtons();
          });
      }
    });
}

void CalibrationPanel::onLoadSession()
{
  if (!serviceReady(*load_client_)) {return;}
  const QString path = load_path_->text().trimmed();
  if (path.isEmpty()) {
    setMessage("enter a session directory (absolute or relative to data_root)", true);
    return;
  }
  auto req = std::make_shared<cec::srv::LoadSession::Request>();
  req->session_dir = path.toStdString();
  pending_ = true;
  updateButtons();
  std::weak_ptr<bool> alive = alive_;
  load_client_->async_send_request(req,
    [this, alive](rclcpp::Client<cec::srv::LoadSession>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            updateButtons();
          });
      }
    });
}

void CalibrationPanel::onToggleManual()
{
  if (!serviceReady(*manual_client_)) {return;}
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = status_.robot_mode != 2;
  pending_ = true;
  updateButtons();
  setMessage(req->data ? "switching to manual mode..." : "switching to position mode...");
  std::weak_ptr<bool> alive = alive_;
  manual_client_->async_send_request(req,
    [this, alive](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture f) {
      auto res = f.get();
      if (auto a = alive.lock(); a && *a) {
        postToGui([this, res] {
            pending_ = false;
            setMessage(QString::fromStdString(res->message), !res->success);
            updateButtons();
          });
      }
    });
}

// ------------------------------------------------------------------ rviz config
void CalibrationPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("columns", columns_->value());
  config.mapSetValue("rows", rows_->value());
  config.mapSetValue("square_size_mm", square_->value());
  config.mapSetValue("mode", mode_->currentData().toString());
  config.mapSetValue("method", method_->currentText());
}

void CalibrationPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  int i;
  float f;
  QString s;
  if (config.mapGetInt("columns", &i)) {columns_->setValue(i);}
  if (config.mapGetInt("rows", &i)) {rows_->setValue(i);}
  if (config.mapGetFloat("square_size_mm", &f)) {square_->setValue(f);}
  if (config.mapGetString("mode", &s)) {
    const int idx = mode_->findData(s);
    if (idx >= 0) {mode_->setCurrentIndex(idx);}
  }
  if (config.mapGetString("method", &s)) {
    const int idx = method_->findText(s);
    if (idx >= 0) {method_->setCurrentIndex(idx);}
  }
}

}  // namespace camera_extrinsic_calibration_rviz

PLUGINLIB_EXPORT_CLASS(camera_extrinsic_calibration_rviz::CalibrationPanel, rviz_common::Panel)
