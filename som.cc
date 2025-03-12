#define RAYGUI_IMPLEMENTATION
#include <raygui.h>
#include <raylib.h>
#include <raymath.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <random>

// Generates a random double in range [min, max).
double RandomDouble(double min, double max) {
  static std::random_device random_device;
  static std::mt19937 random_engine(random_device());
  std::uniform_real_distribution<> dist(min, max);
  return dist(random_engine);
}

/*******************************************************************************
 * Vec
 ******************************************************************************/

class Vec {
 public:
  using SizeT = std::array<double, 3>::size_type;

 private:
  std::array<double, 3> data_;

 public:
  Vec() : data_{0.0, 0.0, 0.0} {}
  Vec(double r, double g, double b) : data_{r, g, b} {}
  Vec(const Vec&) = default;

  Vec& operator+=(const Vec& other) {
    for (SizeT i = 0; i < data_.size(); ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }

  Vec& operator-=(const Vec& other) {
    for (SizeT i = 0; i < data_.size(); ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }

  Vec& operator*=(double scalar) {
    for (SizeT i = 0; i < data_.size(); ++i) {
      data_[i] *= scalar;
    }
    return *this;
  }

  double& operator[](SizeT index) { return data_[index]; }

  const double& operator[](SizeT index) const { return data_[index]; }

  void Randomize(double min, double max) {
    for (SizeT i = 0; i < data_.size(); ++i) {
      data_[i] = RandomDouble(min, max);
    }
  }

  friend double EuclideanDistance(const Vec&, const Vec&);
};

Vec operator+(Vec lhs, const Vec& rhs) { return lhs += rhs; }
Vec operator-(Vec lhs, const Vec& rhs) { return lhs -= rhs; }
Vec operator*(Vec lhs, double scalar) { return lhs *= scalar; }
Vec operator*(double scalar, Vec rhs) { return rhs *= scalar; }

double EuclideanDistance(const Vec& lhs, const Vec& rhs) {
  double sum = 0.0;
  for (Vec::SizeT i = 0; i < lhs.data_.size(); ++i) {
    sum += std::pow(lhs.data_[i] - rhs.data_[i], 2);
  }
  return std::sqrt(sum);
}

/*******************************************************************************
 * Hex
 ******************************************************************************/

struct Hex {
  const int q, r, s;

  Hex(int q, int r, int s) : q(q), r(r), s(s) { assert(q + r + s == 0); }
  Hex(const Hex&) = default;
};

unsigned HexDistance(const Hex& lhs, const Hex& rhs) {
  auto diff = Hex(lhs.q - rhs.q, lhs.r - rhs.r, lhs.s - rhs.s);
  return unsigned(
      std::max({std::abs(diff.q), std::abs(diff.r), std::abs(diff.s)}));
}

std::vector<Hex> GenerateHexGrid(int rows_top, int rows_bottom, int cols_left,
                                 int cols_right) {
  std::vector<Hex> grid;

  for (int q = cols_left; q <= cols_right; ++q) {
    int q_offset = q >> 1;
    for (int r = rows_top - q_offset; r <= rows_bottom - q_offset; ++r) {
      grid.push_back(Hex(q, r, -q - r));
    }
  }

  return grid;
}

/*******************************************************************************
 * Node
 ******************************************************************************/

struct Node {
  const Hex hex;
  Vec weight;

  Node(const Hex& hex) : hex(hex), weight() {}
  Node(const Node&) = default;
};

/*******************************************************************************
 * Som
 ******************************************************************************/

class Som {
 public:
  using NodesT = std::vector<Node*>;
  static constexpr unsigned kTrainingEpochLimit = 10;

 private:
  NodesT nodes_;
  unsigned current_training_epoch_;

 public:
  Som(int rows, int cols) {
    int rows_bottom = rows / 2;
    int rows_top = -(rows - rows_bottom - 1);
    int cols_right = cols / 2;
    int cols_left = -(cols - cols_right - 1);
    auto hexes = GenerateHexGrid(rows_top, rows_bottom, cols_left, cols_right);

    for (auto hex : hexes) {
      nodes_.push_back(new Node(hex));
      nodes_.back()->weight.Randomize(0.0, 255.0);
    }

    current_training_epoch_ = 0;
  }

  ~Som() {
    for (auto node : nodes_) {
      delete node;
    }
  }

  const NodesT& GetNodes() const { return nodes_; }

  bool TrainOneEpoch(const std::vector<Vec>& input_batch) {
    ++current_training_epoch_;
    if (current_training_epoch_ > kTrainingEpochLimit) {
      return false;
    }

    double learning_rate = 1 - (static_cast<double>(current_training_epoch_) /
                                static_cast<double>(kTrainingEpochLimit));

    for (auto input_vec : input_batch) {
      Node* bmu = FindBmu(input_vec);
      assert(bmu);
      UpdateWeights(bmu, input_vec, learning_rate);
    }

    return true;
  }

  NodesT SaveSnapshot() {
    NodesT snapshot;
    for (auto node : nodes_) {
      snapshot.push_back(new Node(*node));
    }
    return snapshot;
  }

 private:
  Node* FindBmu(const Vec& input_vec) const {
    Node* bmu = nullptr;
    for (auto& node : nodes_) {
      if ((bmu == nullptr) || (EuclideanDistance(input_vec, node->weight) <
                               EuclideanDistance(input_vec, bmu->weight))) {
        bmu = node;
      }
    }
    return bmu;
  }

  void UpdateWeights(Node* bmu, const Vec& input_vec, double learning_rate) {
    for (auto& node : nodes_) {
      double neighborhood_coeff =
          1 / std::pow(2, HexDistance(bmu->hex, node->hex));
      node->weight +=
          neighborhood_coeff * learning_rate * (input_vec - node->weight);
    }
  }
};

/*******************************************************************************
 * UI layout consts
 ******************************************************************************/

constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;
constexpr Vector2 kWindowMiddle = Vector2(kWindowWidth / 2, kWindowHeight / 2);
constexpr Vector2 kSidebarPos = Vector2(0, 0);
constexpr Vector2 kSidebarSize = Vector2(kWindowWidth / 5, kWindowHeight);
constexpr Vector2 kHexSize = Vector2(70, 70);
constexpr Vector2 kMapOrigin = Vector2(kWindowMiddle.x + kSidebarSize.x / 2,
                                       kWindowMiddle.y - kHexSize.y / 2);

/*******************************************************************************
 * HexGridDrawer
 ******************************************************************************/

class HexGridDrawer {
 private:
  Vector2 origin_;
  Vector2 size_;

 public:
  HexGridDrawer(Vector2 origin, Vector2 size) : origin_(origin), size_(size) {}

  bool DrawHex(const Hex& hex, Color color) {
    auto center = ToPixel(hex);
    auto corners = GetCorners(hex);

    bool hex_is_selected = false;

    for (int i = 0; i < 6; ++i) {
      auto corner_a = corners[i];
      auto corner_b = corners[(i + 1) % 6];
      DrawTriangle(center, corner_b, corner_a, color);

      Vector2 mouse_pos = Vector2(GetMouseX(), GetMouseY());
      if (!hex_is_selected) {
        hex_is_selected =
            CheckCollisionPointTriangle(mouse_pos, center, corner_a, corner_b);
      }
    }

    return hex_is_selected;
  }

  void HighlightHex(const Hex& hex) {
    auto corners = GetCorners(hex);

    for (int i = 0; i < 6; ++i) {
      auto corner_a = corners[i];
      auto corner_b = corners[(i + 1) % 6];
      DrawLineEx(corner_a, corner_b, 5.0, BLACK);
    }
  }

 private:
  Vector2 ToPixel(const Hex& hex) const {
    static std::array<double, 4> f = {3.0 / 2.0, 0.0, std::sqrt(3.0) / 2.0,
                                      std::sqrt(3.0)};
    double x = (f[0] * hex.q + f[1] * hex.r) * size_.x + origin_.x;
    double y = (f[2] * hex.q + f[3] * hex.r) * size_.y + origin_.y;
    return Vector2(x, y);
  }

  Vector2 CornerOffset(int corner) const {
    double angle = 2.0 * std::numbers::pi * corner / 6;
    return Vector2(size_.x * std::cos(angle), size_.y * std::sin(angle));
  }

  std::vector<Vector2> GetCorners(const Hex& hex) const {
    std::vector<Vector2> corners;
    Vector2 center = ToPixel(hex);
    for (int i = 0; i < 6; ++i) {
      Vector2 offset = CornerOffset(i);
      corners.push_back(Vector2Add(center, offset));
    }
    return corners;
  }
};

/*******************************************************************************
 * SidebarDrawer
 ******************************************************************************/

class SidebarDrawer {
 private:
  static constexpr float row_height_ = 70.0;
  static constexpr float padding_ = 20.0;
  const Vector2 pos_;
  const Vector2 size_;

 public:
  SidebarDrawer(const Vector2& pos, const Vector2& size)
      : pos_(Vector2AddValue(pos, padding_)),
        size_(Vector2SubtractValue(size, 2 * padding_)) {}

  void Draw(unsigned& epoch, Node* selected_node) {
    DrawRectangleV(Vector2SubtractValue(pos_, padding_),
                   Vector2AddValue(size_, 2 * padding_), LIGHTGRAY);

    int row = 0;
    DrawLabel(row++, "Epoch " + std::to_string(epoch));
    DrawEpochButtons(row++, epoch);
    if (selected_node != nullptr) {
      row++;

      auto hex = selected_node->hex;
      DrawLabel(row++, "Node (" + std::to_string(hex.q) + "," +
                           std::to_string(hex.r) + ")");

      auto weight = selected_node->weight;
      DrawLabel(row++, "\tr: " + std::to_string(weight[0]));
      DrawLabel(row++, "\tg: " + std::to_string(weight[1]));
      DrawLabel(row++, "\tb: " + std::to_string(weight[2]));
    }
  }

 private:
  void DrawLabel(int row, std::string&& text) {
    Rectangle pos = {pos_.x, pos_.y + row * row_height_, size_.x, row_height_};
    GuiLabel(pos, text.c_str());
  }

  void DrawEpochButtons(int row, unsigned& value) {
    float width = size_.x / 2;
    Rectangle prev_pos = {pos_.x, pos_.y + row * row_height_, width,
                          row_height_};
    Rectangle next_pos = {pos_.x + width, pos_.y + row * row_height_, width,
                          row_height_};
    if (GuiButton(prev_pos, "-1")) {
      if (value > 0) {
        value -= 1;
      }
    } else if (GuiButton(next_pos, "+1")) {
      if (value < Som::kTrainingEpochLimit) {
        value += 1;
      }
    }
  }
};

/*******************************************************************************
 * Main
 ******************************************************************************/

void SetUpRaylibWindow(int window_width, int window_height) {
  InitWindow(window_width, window_height, "SOM");
  SetTargetFPS(30);
  GuiSetStyle(DEFAULT, TEXT_SIZE, 40);
  SetTextureFilter(GetFontDefault().texture, ICON_FILTER_POINT);
}

std::vector<Vec> GenerateTrainingData() {
  std::vector<Vec> data;
  for (double i = 0; i < 2; ++i) {
    data.push_back({255.0, 0.0, 0.0});
    data.push_back({255.0, 127.0, 0.0});
    data.push_back({255.0, 255.0, 0.0});
    data.push_back({0.0, 255.0, 0.0});
    data.push_back({0.0, 0.0, 255.0});
    data.push_back({75.0, 0.0, 130.0});
    data.push_back({148.0, 0.0, 211.0});
  }
  return data;
}

int main(void) {
  SetUpRaylibWindow(kWindowWidth, kWindowHeight);
  HexGridDrawer hex_grid(kMapOrigin, kHexSize);
  SidebarDrawer sidebar(kSidebarPos, kSidebarSize);

  auto som = Som(7, 11);
  auto training_data = GenerateTrainingData();

  std::vector<Som::NodesT> epoch_snapshots;
  do {
    epoch_snapshots.push_back(som.SaveSnapshot());
  } while (som.TrainOneEpoch(training_data));

  unsigned selected_epoch = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    Node* selected_node = nullptr;
    for (const auto& node : epoch_snapshots[selected_epoch]) {
      Color color = {static_cast<unsigned char>(node->weight[0]),
                     static_cast<unsigned char>(node->weight[1]),
                     static_cast<unsigned char>(node->weight[2]), 255};
      if (hex_grid.DrawHex(node->hex, color)) {
        selected_node = node;
      }
    }

    if (selected_node) {
      hex_grid.HighlightHex(selected_node->hex);
    }

    sidebar.Draw(selected_epoch, selected_node);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
