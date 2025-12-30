#ifndef FLOODFILLLABEL_H
#define FLOODFILLLABEL_H

#include <vector>
#include <queue>
#include <utility>
#include "Region.h"

std::pair<std::vector<int>, std::vector<Region>>
floodFillLabel(const uint8_t* const image, const int width, const int height);

#endif // FLOODFILLLABEL_H
