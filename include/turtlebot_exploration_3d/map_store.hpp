#pragma once
#include <octomap/octomap.h>
#include <shared_mutex>
#include <memory>
#include <atomic>

namespace explore {

using point3d = octomap::point3d;

struct MapStore {
  // writer는 live에 적재, 주기적으로 snapshot으로 복제
  std::shared_ptr<octomap::OcTree> live;
  std::shared_ptr<octomap::OcTree> snapshot;

  // 센서 원점 (live / snapshot 시점)
  point3d live_origin{0,0,0};
  point3d snap_origin{0,0,0};

  // 동기화
  std::shared_mutex mtx;      // snapshot 교체 보호
  std::atomic<uint64_t> stamp{0};

  // 공통 파라미터
  double max_range{6.0};
  double resolution{0.05};
};

// 전역 스토어 (동일 프로세스 내 컴포넌트 공유)
inline MapStore& global_map_store() {
  static MapStore s;
  return s;
}

} // namespace explore
