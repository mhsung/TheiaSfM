// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_drawing_utils.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <glog/logging.h>
#include <stlplus3/file_system.hpp>


namespace theia {
void hsv2rgb(
    const float h, const float s, const float v,
    float& r, float& g, float& b)
{
  float hh, p, q, t, ff;
  long i;

  if (s <= 0.0f) {
    r = v; g = v; b = v;
    return;
  }

  hh = h * 360.0f;
  if (hh >= 360.0f) hh = 0.0f;
  hh /= 60.0f;
  i = (long)hh;
  ff = hh - i;
  p = v * (1.0f - s);
  q = v * (1.0f - (s * ff));
  t = v * (1.0f - (s * (1.0f - ff)));

  switch (i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
    default:
      r = v; g = p; b = q;
      break;
  }
}

theia::RGBPixel LabelColor(const uint32_t label)
{
  // Reference:
  // http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
  const float golden_ratio_conjugate = 0.618033988749895f;
  const float start_h = 0.5f;
  const float h = std::fmod(
      start_h + static_cast<float>(label) * golden_ratio_conjugate, 1.0f);
  float r, g, b;
  hsv2rgb(h, 0.9f, 0.9f, r, g, b);
  return theia::RGBPixel(255.0f * r, 255.0f * g, 255.0f * b);
}


void DrawBox(const Eigen::Vector4d& box, const theia::RGBPixel& color,
             theia::ImageCanvas* canvas) {
  CHECK_NOTNULL(canvas);
  const int x1 = static_cast<int>(std::round(box[0]));
  const int y1 = static_cast<int>(std::round(box[1]));
  const int x2 = static_cast<int>(std::round(box[2]));
  const int y2 = static_cast<int>(std::round(box[3]));
  canvas->DrawLine(x1, y1, x2, y1, color);
  canvas->DrawLine(x1, y1, x1, y2, color);
  canvas->DrawLine(x1, y2, x2, y2, color);
  canvas->DrawLine(x2, y1, x2, y2, color);
}
}