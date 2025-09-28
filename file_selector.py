"""
文件选择器 - file_selector.py
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional


class FileSelector:
    """文件选择器工具类"""
    
    @staticmethod
    def find_h5_files(folder_paths: List[str], max_depth: int = 3) -> List[Tuple[str, str]]:
        """
        在多个文件夹及其子目录中查找.h5文件
        
        Args:
            folder_paths: 文件夹路径列表
            max_depth: 最大检索深度
            
        Returns:
            [(相对路径, 绝对路径)] 的列表，按字母顺序排序
        """
        all_h5_files = []
        
        for folder_path in folder_paths:
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                print(f"警告: 文件夹路径不存在: {folder_path}")
                continue
            
            if not os.path.isdir(folder_path):
                print(f"警告: 路径不是文件夹: {folder_path}")
                continue

            base_depth = folder_path.rstrip(os.sep).count(os.sep)
            folder_name = os.path.basename(os.path.normpath(folder_path))

            # 遍历文件夹（限制深度）
            for root, dirs, files in os.walk(folder_path):
                current_depth = root.count(os.sep) - base_depth
                if current_depth >= max_depth:
                    dirs[:] = []  # 不再深入
                    continue
                    
                for file in files:
                    if file.endswith(".h5"):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, folder_path)
                        # 在相对路径前加上顶层目录名，避免冲突
                        combined_relpath = os.path.join(folder_name, relative_path)
                        all_h5_files.append((combined_relpath, full_path))

        # 按相对路径排序
        all_h5_files.sort(key=lambda x: x[0])
        
        return all_h5_files
    
    @staticmethod
    def select_file_interactive(file_list: List[Tuple[str, str]]) -> Optional[str]:
        """
        交互式选择文件
        
        Args:
            file_list: [(相对路径, 绝对路径)] 的列表
            
        Returns:
            选中的文件绝对路径，如果用户取消则返回None
        """
        if not file_list:
            print("没有找到.h5文件")
            return None
        
        # 显示文件列表
        print(f"找到 {len(file_list)} 个.h5文件:")
        print("-" * 60)
        
        for i, (rel_path, _) in enumerate(file_list, 1):
            print(f"{i:3d}. {rel_path}")
        
        print("-" * 60)
        
        # 获取用户输入
        while True:
            try:
                choice = input("请选择文件编号 (输入q退出): ").strip()
                
                if choice.lower() == 'q':
                    print("用户选择退出")
                    return None
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(file_list):
                    rel_path, full_path = file_list[choice_num - 1]
                    print(f"已选择: {rel_path}")
                    return full_path
                else:
                    print(f"请输入 1-{len(file_list)} 之间的数字")
                    
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断操作")
                return None
    
    @staticmethod
    def select_h5_file(folder_paths, max_depth: int = 3) -> Optional[str]:
        """
        在多个文件夹中查找并选择.h5文件
        
        Args:
            folder_paths: 字符串或字符串列表，文件夹路径
            max_depth: 最大检索深度
            
        Returns:
            选择的.h5文件的绝对路径
        """
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
        
        # 查找文件
        file_list = FileSelector.find_h5_files(folder_paths, max_depth)
        
        # 交互式选择
        return FileSelector.select_file_interactive(file_list)
    
    @staticmethod
    def batch_select_files(folder_paths, max_depth: int = 3) -> List[str]:
        """
        批量选择多个文件
        
        Args:
            folder_paths: 文件夹路径
            max_depth: 最大检索深度
            
        Returns:
            选中的文件路径列表
        """
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
        
        file_list = FileSelector.find_h5_files(folder_paths, max_depth)
        selected_files = []
        
        if not file_list:
            print("没有找到.h5文件")
            return selected_files
        
        print(f"找到 {len(file_list)} 个.h5文件:")
        print("可以选择多个文件，用逗号分隔，例如: 1,3,5")
        print("-" * 60)
        
        for i, (rel_path, _) in enumerate(file_list, 1):
            print(f"{i:3d}. {rel_path}")
        
        print("-" * 60)
        
        while True:
            try:
                choice = input("请选择文件编号 (输入q退出, a选择全部): ").strip()
                
                if choice.lower() == 'q':
                    break
                elif choice.lower() == 'a':
                    selected_files = [full_path for _, full_path in file_list]
                    print(f"已选择全部 {len(selected_files)} 个文件")
                    break
                else:
                    # 解析用户输入的编号
                    choices = [int(x.strip()) for x in choice.split(',')]
                    valid_choices = [c for c in choices if 1 <= c <= len(file_list)]
                    
                    if valid_choices:
                        selected_files = [file_list[i-1][1] for i in valid_choices]
                        print(f"已选择 {len(selected_files)} 个文件:")
                        for i in valid_choices:
                            print(f"  - {file_list[i-1][0]}")
                        break
                    else:
                        print("请输入有效的编号")
                        
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断操作")
                break
        
        return selected_files