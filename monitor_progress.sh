#!/bin/bash

echo "🌲 МОНИТОРИНГ МЕГА-СИСТЕМЫ 19 ВИДОВ 🌲"
echo "========================================"

while true; do
    clear
    echo "🌲 МОНИТОРИНГ МЕГА-СИСТЕМЫ 19 ВИДОВ 🌲"
    echo "========================================"
    echo "⏰ $(date '+%H:%M:%S')"
    echo ""
    
    # Проверка процесса
    if pgrep -f mega_system_19_species > /dev/null; then
        echo "✅ СТАТУС: РАБОТАЕТ"
        
        # Статистика процесса
        ps aux | grep mega_system_19_species | grep -v grep | while read line; do
            echo "📊 CPU: $(echo $line | awk '{print $3}')%"
            echo "💾 RAM: $(echo $line | awk '{print $4}')%"
            echo "⏱️  Время: $(echo $line | awk '{print $10}')"
        done
        
        echo ""
        echo "📋 ПОСЛЕДНИЕ СТРОКИ ЛОГА:"
        echo "------------------------"
        tail -n 5 mega_19_log.txt 2>/dev/null || echo "Лог еще не создан..."
        
    else
        echo "🏁 ПРОЦЕСС ЗАВЕРШЕН!"
        
        if [ -f mega_19_log.txt ]; then
            echo ""
            echo "📋 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:"
            echo "----------------------"
            tail -n 20 mega_19_log.txt
        fi
        
        # Проверка созданных файлов
        echo ""
        echo "📁 СОЗДАННЫЕ ФАЙЛЫ:"
        echo "-----------------"
        ls -la mega_19_species_* 2>/dev/null || echo "Файлы не найдены"
        ls -la *19_species*.png 2>/dev/null || echo "Графики не найдены"
        
        break
    fi
    
    echo ""
    echo "🔄 Обновление через 30 секунд... (Ctrl+C для выхода)"
    sleep 30
done 