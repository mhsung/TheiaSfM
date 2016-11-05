// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

class JsonFile {
  public:
    bool Open(const std::string& filename) {
      file.open(filename);
      if (!file.is_open()) return false;
      is_first_element_added = false;
      return true;
    }

    bool IsOpen() const { return file.is_open(); }

    template <class T>
    void WriteElement(const std::string& name, const T& value) {
      CHECK (file.is_open()) << " Open json file first.";
      if (!is_first_element_added) {
        file << "{" << std::endl;
        is_first_element_added = true;
      } else {
        file << "," << std::endl;
      }
      file << "  \"" << name << "\": " << value;
    }

    void Close() {
      if (file.is_open()) {
        file << std::endl;
        file << "}" << std::endl;
        file.close();
      }
    }

  private:
    std::ofstream file;
    bool is_first_element_added;
};
