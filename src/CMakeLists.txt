find_package(CUDA)
find_package(CUDAToolkit)
find_package(cuml) # ML:DBScan
find_package(cugraph) # for graph

# message(STATUS "Cugraph DIR" ${CUGRAPH_INCLUDE_DIRS})
# add_executable(weakly_connected_components_test weakly_connected_components_test.cpp)

# target_link_libraries(weakly_connected_components_test
# PUBLIC
#     cugraph
# )



# add_executable(dbscan_test cugraphExample.cu)

# # set_target_properties(cugraph_example
# #     PROPERTIES CXX_STANDARD                        17
# #                CXX_STANDARD_REQUIRED               ON
# #                CUDA_STANDARD                       17
# #                CUDA_STANDARD_REQUIRED              ON
# #                INTERFACE_POSITION_INDEPENDENT_CODE ON
# # )


# target_include_directories(cugraph_example 
#     PRIVATE 
#     ${CUB_INCLUDE_DIR}
#     ${CUCO_INCLUDE_DIR}
#     ${CUML_INCLUDE_DIRS} 
#     ${CUGRAPH_INCLUDE_DIRS} 
#     ${CUDA_INCLUDE_DIRS}  
#     ${RMM_INCLUDE_DIRS} 
#     ${SLOG_INCLUDE_DIRS})
# set_target_properties(cugraph_example PROPERTIES LINKER_LANGUAGE "CUDA")
# target_link_libraries(cugraph_example cuml++ cudart cublas cusolver cusparse cugraph cuco)