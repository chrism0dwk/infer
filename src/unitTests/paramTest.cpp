/*
 * paramTest.cpp
 *
 *  Created on: Oct 20, 2010
 *      Author: stsiab
 */

#include <iostream>
#include <ctime>
#include <math.h>

#include <Parameter.hpp>

using namespace std;
using namespace EpiRisk;

struct MyPrior : public Prior
{
public:
  MyPrior() {}
};


int
main(int argc, char* argv[])
{
  // Tests out the Parameter class

  Parameter alpha(2.5,UniformPrior());
  Parameter beta(3.0,UniformPrior());

  clock_t start,end;
  double ans;
  start = clock();
  for(size_t i=0; i<1e9; ++i) {
      ans = alpha + beta;
  }
  end = clock();

  double classTime = end - start;

  double a = 2.5;
  double b = 3.0;

  start = clock();
  for(size_t i=0; i<1e9; ++i) {
      ans = a + b;
  }
  end = clock();

  double baseTime = end - start;

  cout << "Class time: " << classTime << "\n";
  cout << "Base type time: " << baseTime << "\n";
  cout << "Speedup (class/base): x" << (double)classTime/(double)baseTime << endl;



  return EXIT_SUCCESS;

}
