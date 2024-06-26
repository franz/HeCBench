HF_HOME := ../halo_finder
#HF_TYPE_FLAGS := -DID_32 -DPOSVEL_32 -DGRID_32
#HF_TYPE_FLAGS := -DID_64 -DPOSVEL_64 -DGRID_64
HF_TYPE_FLAGS := -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER
HF_HEADERS := ${HF_HOME}/Definition.h
HF_HEADERS += ${HF_HOME}/Partition.h
HF_HEADERS += ${HF_HOME}/ParticleExchange.h
HF_HEADERS += ${HF_HOME}/InitialExchange.h
HF_HEADERS += ${HF_HOME}/GridExchange.h
HF_HEADERS += ${HF_HOME}/ParticleDistribute.h
HF_HEADERS += ${HF_HOME}/CosmoHaloFinderP.h
HF_HEADERS += ${HF_HOME}/FOFHaloProperties.h
#HF_WARNING := -Wmissing-noreturn -Wunused -Wsign-compare -Wshadow -Wformat
HF_CFLAGS := $(EXTRA_CFLAGS) -I${HF_HOME} ${HF_TYPE_FLAGS} ${HF_WARNING}
HF_CXXFLAGS := -I${HF_HOME} ${HF_TYPE_FLAGS} ${HF_WARNING}
HF_LDFLAGS := -L${HF_HOME}/${HACC_OBJDIR}
