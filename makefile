################################################################################

# Target rules
METHODS = label_equivalence_direct_2D_clamp playne_equivalence_direct_2D_clamp playne_equivalence_block_2D_clamp label_equivalence_direct_3D_clamp playne_equivalence_direct_3D_clamp playne_equivalence_block_3D_clamp

NVCC = nvcc -O3
INCLUDE = -I./include/
ARCH = -arch=sm_35 \
       -gencode=arch=compute_35,code=sm_35 \
       -gencode=arch=compute_50,code=sm_50 \
       -gencode=arch=compute_52,code=sm_52 \
       -gencode=arch=compute_60,code=sm_60 \
       -gencode=arch=compute_61,code=sm_61 \
       -gencode=arch=compute_70,code=sm_70 \
       -gencode=arch=compute_75,code=sm_75 \
       -gencode=arch=compute_75,code=compute_75

all: $(METHODS)

%: %.o
	$(NVCC) $(ARCH) $(INCLUDE) $< -o $@

%.o: %.cu
	$(NVCC) $(ARCH) $(INCLUDE) -c $< -o $@

# Clean
clean:
	rm $(METHODS) *.o