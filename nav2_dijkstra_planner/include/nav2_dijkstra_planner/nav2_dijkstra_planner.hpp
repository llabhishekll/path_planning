#ifndef NAV2_DIJKSTRA_PLANNER__DIJKSTRA_LINE_PLANNER_HPP_
#define NAV2_DIJKSTRA_PLANNER__DIJKSTRA_LINE_PLANNER_HPP_

#include <memory>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>

#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav_msgs/msg/path.hpp"

namespace nav2_dijkstra_planner {

/**

@class DijkstraPlanner
@brief A global planner that implements Dijkstra's algorithm for finding the
shortest path between two points and forwards it to the move_base global
planner module
*/
class DijkstraGlobalPlanner : public nav2_core::GlobalPlanner {
public:
  /**
  @brief Constructor for the DijkstraGlobalPlanner
  */
  DijkstraGlobalPlanner() = default;
  /**

  @brief Destructor for the DijkstraGlobalPlanner
  */
  ~DijkstraGlobalPlanner() = default;

  // plugin configure
  void configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent, std::string name,
      std::shared_ptr<tf2_ros::Buffer> tf,
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  // plugin cleanup
  void cleanup() override;

  // plugin activate
  void activate() override;

  // plugin deactivate
  void deactivate() override;

  /**

  @brief Given a goal pose in the world, compute a plan
  @param start The start pose
  @param goal The goal pose
  @return The plan filled by the planner as a Path
  */
  nav_msgs::msg::Path
  createPlan(const geometry_msgs::msg::PoseStamped &start,
             const geometry_msgs::msg::PoseStamped &goal) override;

private:
  int i;

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  // std::shared_ptr<nav2_util::NodeUtilities> node_util_;
  /** @brief Points to occupacy grid (costmap) with ROS functionallity */
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  double interpolation_resolution_;
  // The global frame of the costmap
  std::string global_frame_, name_;

  /**
  @brief Performs Dijkstra's shortest path algorithm search on a costmap
  @param start_cell_index index value corresponding to the start cell
  location on a one-dimensional array map
  @param goal_cell_index index value corresponding to the goal cell location
  on a one-dimensional array map
  @param costmap_flat values in the occupacy grid as a flat array map
  representation
  @param shortest_path keeps the vector indices that correspond to the grid
  cells that give the shortest path to the goal
  @return true on success, false if an error occurred
  */
  bool dijkstraShortestPath(const int &start_cell_index,
                            const int &goal_cell_index,
                            const std::vector<int> &costmap_flat,
                            std::vector<int> &shortest_path);

  /**
   * @brief Checks if world coordinates are inside grid map bounds
   * @param x X-Axis value in the world frame of reference (in meters)
   * @param y Y-Axis value in the world frame of reference (in meters)
   * @return true if index is in map bounds, otherwise false
   */
  bool inGridMapBounds(const float &x, const float &y);

  /**
   * @brief Converts x,y values in the world frame (in meters) into x,y grid map
   * coordinates This transformation is derived from the map resolution and
   * adjusts w.r.t the location of the map origin
   * @param x X-Axis value in the world frame of reference (in meters)
   * @param y Y-Axis value in the world frame of reference (in meters)
   */
  void fromWorldToGrid(float &x, float &y);

  /**
   * @brief Converts x,y grid cell coordinates to world coordinates (in meters)
   *        This transformation is derived from the map resolution, adjusts
   *        w.r.t the location of the map origin and can include an offset
   *        to place the world coordinate at the center point of a grid cell
   * @param x Grid cell map x coordinate value
   * @param y Grid cell map y coordinate value
   */
  void fromGridToWorld(float &x, float &y);

  /**
   * @brief Converts a x,y grid cell coordinate value to a linear index value
   * (one-dimensional array index)
   * @param x Grid cell map x coordinate value
   * @param y Grid cell map y coordinate value
   * @return index value corresponding to the location on the one-dimensional
   * array representation
   */
  size_t gridCellxyToIndex(const float &x, const float &y);

  /**
   * @brief Converts a linear index value to a x,y grid cell coordinate value
   * @param index A linear index value, specifying a cell/pixel in an 1-D array
   * @param x Grid cell map x coordinate value
   * @param y Grid cell map y coordinate value
   */
  void fromIndexToGridCellxy(size_t index, int &x, int &y);

  /**
   * @brief Identifies neighbor nodes inspecting the 8 adjacent nodes
   * @param current_node index value of current grid cell who's neighbor nodes
   * must be determined
   * @return hash table of neighbor node indices and their associated step costs
   */
  std::unordered_map<int, double>
  find_neighbors(const int &current_node, const std::vector<int> &costmap_flat);

  template <typename K, typename V>
  void print_map(std::unordered_map<K, V> const &m) {
    for (auto const &pair : m) {
      std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
  }

  /** @brief Points to the values in the occupacy grid */
  nav2_costmap_2d::Costmap2D *costmap_;
  /** @brief The x value of the origin of the costmap */
  float origin_x_;
  /** @brief The y value of the origin of the costmap */
  float origin_y_;
  /** @brief The resolution of the costmap, equivalent value of 1 pixel of the
   * costmap in meters */
  float resolution_;
  /** @brief The the x size of the costmap in number of grid cells */
  unsigned int width_;
  /** @brief The the y size of the costmap in number of grid cells */
  unsigned int height_;
  /** @brief The total size of the costmap in number of grid cells  */
  unsigned int map_size_;
  /** @brief The initialization status of the instance variables  */
  bool initialized_ = false;
};

} // namespace nav2_dijkstra_planner
#endif