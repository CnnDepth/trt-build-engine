#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

class Logger : public nvinfer1::ILogger
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
} gLogger;

int main( int argc, char** argv )
{

  //LeNet5 test UFF file
  std::string uffFile( "test.uff" );


  nvinfer1::IBuilder*           builder = nvinfer1::createInferBuilder( gLogger );
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  nvuffparser::IUffParser*      parser  = nvuffparser::createUffParser();

  parser->registerInput( "Input_0", nvinfer1::DimsCHW(1, 28, 28), nvuffparser::UffInputOrder::kNCHW );
  parser->registerOutput( "Binary_3" );

  std::cout << "Parsing network...";
  parser->parse( uffFile.c_str(), *network, nvinfer1::DataType::kFLOAT );
  parser->destroy();
  std::cout << "  DONE!" << std::endl;

  //Configuring and building engine
  builder->setMaxBatchSize( 1 );
  builder->setMaxWorkspaceSize( 1 << 20 );

  std::cout << "Building engine...";  
  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine( *network );
  network->destroy();
  builder->destroy();
  std::cout << "  DONE!" << std::endl;

  std::cout << "Serializing model...";
  nvinfer1::IHostMemory* serializedModel = engine->serialize();
  engine->destroy();
  std::cout << "  DONE!" << std::endl;

  std::cout << "Saving serialized model to disk...";
  std::ofstream fileEngine( "test.trt", std::ios::out | std::ios::binary );
  fileEngine.write( (char*)(serializedModel->data()), serializedModel->size() );
  fileEngine.close();
  serializedModel->destroy();
  std::cout << "  DONE!" << std::endl;


  return 0;
}
