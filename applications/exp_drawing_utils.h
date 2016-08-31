// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>


namespace theia {
theia::RGBPixel LabelColor(const uint32_t label);

// @box: xmin, ymin, xmax, ymax.
void DrawBox(const Eigen::Vector4d& box, const theia::RGBPixel& color,
             theia::ImageCanvas* canvas);
}