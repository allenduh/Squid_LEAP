.DEFAULT_GOAL = all

# we use g++ to build 64-bit binaries. No warning is allowed.
CXX=g++
C=gcc
#CFLAG_ARCH = -m64
CFLAGS = -Wall -Werror $(CFLAG_ARCH)
CFLAGS += -fPIE

# all generated objects are at ../out.
DIR_OUT=./out
CXXEXE = $(DIR_OUT)/multistream

# EMERGENT_DIR is /opt/EVT, set by EMERGENT installer
DIR_INC = -I$(EMERGENT_DIR)/eSDK/include/ -I../common/ 
LDLIBS_FLAGS = -L$(EMERGENT_DIR)/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision -lEmergentUtil -lpthread # -g #-ltiff

# compile all c++ files.
SOURCE_CXX = $(wildcard *.cpp)
SOURCE_CXX += $(wildcard framesaver/*.cpp)
OBJS_CXX = $(patsubst %.cpp,$(DIR_OUT)/%.o,$(SOURCE_CXX)) 


# header file dependency
$(DIR_OUT)/%.d: %.cpp
	mkdir -p $(dir $@)
	@set -e; rm -f $@; \
	$(CXX) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(DIR_OUT)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
SOURCE_DEPS = $(patsubst %.cpp,$(DIR_OUT)/%.d,$(SOURCE_CXX))
include $(SOURCE_DEPS)

$(DIR_OUT)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $(CFLAGS) $(DIR_INC) $< -o $@

$(CXXEXE) : $(OBJS_CXX)
	$(CXX) -o $(CXXEXE) $(OBJS_CXX) $(LDLIBS_FLAGS)
	mkdir -p $(EMERGENT_DIR)/eSDK/tools
	cp $(CXXEXE) $(EMERGENT_DIR)/eSDK/tools


.PHONY:all
all:  $(CXXEXE)

.PHONY:clean
clean:
	rm -fr $(DIR_OUT)

