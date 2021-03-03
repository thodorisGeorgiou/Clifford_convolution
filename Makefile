#Base comands
CC=g++
cuCC=CPATH="$(CPATH)":/usr/local/ nvcc

#Necessary pathds
SOURCE_PATH=sources/
INCLUDE_PATH=-I headers/
TARGET_PATH=cliffordConvolution/objectFiles/

#tensorflow flags
TF_CFLAGS=$(shell python3 sources/getTF_CFLAGS.py)
TF_LFLAGS=$(shell python3 sources/getTF_LFLAGS.py)

#GCC compile flags and options
CFLAGS=-std=c++11 -shared -fopenmp -fPIC $(INCLUDE_PATH)
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart
CC_OPTIONS=-DGOOGLE_CUDA=1

#cuda compile flags and options
NVCC_CFLAGS=-std=c++11 -c -DNDEBUG --expt-relaxed-constexpr -x cu -Xcompiler -fPIC $(INCLUDE_PATH)
NVCC_LFLAGS=
NVCC_OPTIONS=-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -DGOOGLE_CUDA=1

# SOURCE_FILES=gather_angles.cc

SOURCE_FILES=gather_angles.cc conv_by_index_2d.cc conv_by_index_input_grads.cc conv_by_index_weight_grads.cc \
			reduce_index.cc expand_index.cc offset_correct.cc weight_to_angle_gradients.cc \
			pool_by_index.cc upsample_index.cc
SOURCES=$(addprefix $(SOURCE_PATH), $(SOURCE_FILES))
CU_OBJECTS=$(SOURCES:.cc=.cu.o)
OBJECTS=$(addprefix $(TARGET_PATH), $(SOURCE_FILES:.cc=.so))



all: $(OBJECTS)

print:
	@echo

$(TARGET_PATH)%.so: $(SOURCE_PATH)%.cu.o $(SOURCE_PATH)%.cc
	$(CC) $(CFLAGS) $(TF_CFLAGS) $^ -o $@ $(CC_OPTIONS) $(LDFLAGS) ${TF_LFLAGS}


%.cu.o: %.cu.cc
	$(cuCC) $(NVCC_CFLAGS) $< -o $@ $(NVCC_OPTIONS) ${TF_CFLAGS}


.PHONY: clean
clean:
	rm $(OBJECTS)
