/*
 Navicat Premium Dump SQL

 Source Server         : 3306
 Source Server Type    : MySQL
 Source Server Version : 80400 (8.4.0)
 Source Host           : localhost:3306
 Source Schema         : cow_face_db

 Target Server Type    : MySQL
 Target Server Version : 80400 (8.4.0)
 File Encoding         : 65001

 Date: 11/05/2025 21:06:34
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for cattle
-- ----------------------------
DROP TABLE IF EXISTS `cattle`;
CREATE TABLE `cattle`  (
  `cattle_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `user_id` int NOT NULL,
  PRIMARY KEY (`cattle_id`) USING BTREE,
  INDEX `user_id`(`user_id` ASC) USING BTREE,
  CONSTRAINT `cattle_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of cattle
-- ----------------------------
INSERT INTO `cattle` VALUES (1, '1', 1);
INSERT INTO `cattle` VALUES (2, '2', 1);
INSERT INTO `cattle` VALUES (3, '7', 1);
INSERT INTO `cattle` VALUES (4, '7', 1);

-- ----------------------------
-- Table structure for model_training
-- ----------------------------
DROP TABLE IF EXISTS `model_training`;
CREATE TABLE `model_training`  (
  `training_id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `model_type` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `epochs` int NOT NULL,
  `batch_size` int NOT NULL,
  `learning_rate` float NOT NULL,
  `start_time` datetime NULL DEFAULT NULL,
  `end_time` datetime NULL DEFAULT NULL,
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `map50` float NULL DEFAULT NULL,
  `map50_95` float NULL DEFAULT NULL,
  `precision` float NULL DEFAULT NULL,
  `recall` float NULL DEFAULT NULL,
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL,
  PRIMARY KEY (`training_id`) USING BTREE,
  INDEX `user_id`(`user_id` ASC) USING BTREE,
  CONSTRAINT `model_training_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 9 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of model_training
-- ----------------------------
INSERT INTO `model_training` VALUES (1, 1, 'n', 50, 16, 0.001, '2025-05-11 11:17:16', NULL, 'training', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `model_training` VALUES (2, 1, 'n', 50, 16, 0.001, '2025-05-11 11:19:32', NULL, 'training', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `model_training` VALUES (3, 1, 'n', 50, 16, 0.001, '2025-05-11 11:20:36', NULL, 'failed', NULL, NULL, NULL, NULL, 'No module named \'numpy._core\'');
INSERT INTO `model_training` VALUES (4, 1, 'n', 50, 16, 0.001, '2025-05-11 11:22:18', NULL, 'failed', NULL, NULL, NULL, NULL, 'No module named \'numpy._core\'');
INSERT INTO `model_training` VALUES (5, 1, 'n', 50, 16, 0.001, '2025-05-11 11:22:56', NULL, 'failed', NULL, NULL, NULL, NULL, 'No module named \'numpy._core\'');
INSERT INTO `model_training` VALUES (6, 1, 'n', 50, 16, 0.001, '2025-05-11 12:22:46', NULL, 'training', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `model_training` VALUES (7, 1, 'n', 50, 16, 0.001, '2025-05-11 12:32:51', NULL, 'failed', NULL, NULL, NULL, NULL, 'Dataset \'cowFaceDetection-main/conference paper preparations/dataset.yaml\' error  \nDataset \'cowFaceDetection-main/conference paper preparations/dataset.yaml\' images not found , missing path \'C:\\Users\\Administrator\\Desktop\\zx\\yolo\\datasets\\cowFaceDetection-main\\conference paper preparations\\test\\images\'\nNote dataset download directory is \'C:\\Users\\Administrator\\Desktop\\zx\\yolo\\datasets\'. You can update this in \'C:\\Users\\Administrator\\AppData\\Roaming\\Ultralytics\\settings.yaml\'');
INSERT INTO `model_training` VALUES (8, 1, 'n', 50, 16, 0.001, '2025-05-11 12:52:53', NULL, 'training', NULL, NULL, NULL, NULL, NULL);

-- ----------------------------
-- Table structure for recognition_record
-- ----------------------------
DROP TABLE IF EXISTS `recognition_record`;
CREATE TABLE `recognition_record`  (
  `record_id` int NOT NULL AUTO_INCREMENT,
  `cattle_id` int NOT NULL,
  `timestamp` datetime NULL DEFAULT NULL,
  `result` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`record_id`) USING BTREE,
  INDEX `cattle_id`(`cattle_id` ASC) USING BTREE,
  CONSTRAINT `recognition_record_ibfk_1` FOREIGN KEY (`cattle_id`) REFERENCES `cattle` (`cattle_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 14 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of recognition_record
-- ----------------------------
INSERT INTO `recognition_record` VALUES (1, 1, '2025-05-11 10:56:59', 'detected');
INSERT INTO `recognition_record` VALUES (2, 1, '2025-05-11 10:57:08', 'detected');
INSERT INTO `recognition_record` VALUES (3, 1, '2025-05-11 10:58:04', 'detected');
INSERT INTO `recognition_record` VALUES (4, 1, '2025-05-11 11:01:00', 'detected (conf: 0.84)');
INSERT INTO `recognition_record` VALUES (5, 1, '2025-05-11 11:03:16', 'Detected: Cow (2) (conf: 0.94)');
INSERT INTO `recognition_record` VALUES (6, 1, '2025-05-11 11:03:47', 'Detected: Cow (2) (conf: 0.33)');
INSERT INTO `recognition_record` VALUES (7, 1, '2025-05-11 11:04:03', 'Detected: Cow (7) (conf: 0.44)');
INSERT INTO `recognition_record` VALUES (8, 1, '2025-05-11 11:04:22', 'Detected: Cow (12) (conf: 1.00)');
INSERT INTO `recognition_record` VALUES (9, 1, '2025-05-11 11:50:23', 'Detected: Cow (12) (conf: 1.00)');
INSERT INTO `recognition_record` VALUES (10, 1, '2025-05-11 12:47:32', '检测置信度: 0.77, 识别置信度: 1.00, 类别: Cow (7)');
INSERT INTO `recognition_record` VALUES (11, 1, '2025-05-11 12:49:38', '检测置信度: 0.77, 识别置信度: 1.00, 类别: Cow (7)');
INSERT INTO `recognition_record` VALUES (12, 1, '2025-05-11 12:52:46', '检测置信度: 0.77, 识别置信度: 1.00, 类别: Cow (7)');
INSERT INTO `recognition_record` VALUES (13, 1, '2025-05-11 12:53:05', '检测置信度: 0.77, 识别置信度: 1.00, 类别: Cow (7)');

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  PRIMARY KEY (`user_id`) USING BTREE,
  UNIQUE INDEX `username`(`username` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, '123', 'scrypt:32768:8:1$6PPzV9OxXcZfbOKz$962287c724593409e1ad89355b46d45e295dba595b142ddcbf89a00b29c668b4c5723f6ed959403e7970e9cadbd0c5840df3e027107a584fa94d87de9f963e93');

SET FOREIGN_KEY_CHECKS = 1;
