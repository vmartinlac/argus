
#pragma once

#include <iostream>

#define ABORT(msg) ( std::cout << __FILE__ << ':' << __LINE__ << ':' << msg << std::endl, exit(1) )

