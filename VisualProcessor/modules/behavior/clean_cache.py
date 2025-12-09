import subprocess
import os

def clear_cache():
    """Очистить кэш памяти"""
    if os.geteuid() != 0:
        print("Требуются права суперпользователя!")
        return False
    
    try:
        # Синхронизируем диски
        subprocess.run(['sync'], check=True)
        
        # Очищаем кэш (1=page cache, 2=dentries+inodes, 3=всё)
        subprocess.run(
            ['sysctl', '-w', 'vm.drop_caches=3'], 
            check=True
        )
        print("Кэш очищен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка: {e}")
        return False

# Использование
if __name__ == "__main__":
    clear_cache()