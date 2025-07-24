#!/bin/bash

echo "🌲🚀 МОНИТОРИНГ 1D ALEXNET ДЛЯ 20 ВИДОВ 🚀🌲"
echo "============================================="

while true; do
    clear
    echo "🌲🚀 МОНИТОРИНГ 1D ALEXNET ДЛЯ 20 ВИДОВ 🚀🌲"
    echo "============================================="
    echo "⏰ $(date '+%H:%M:%S')"
    echo ""
    
    # Проверяем процесс
    if pgrep -f alexnet_20_species > /dev/null; then
        echo "✅ СТАТУС: ALEXNET ОБУЧАЕТСЯ!"
        
        # Показываем CPU и RAM
        ps aux | grep alexnet_20_species | grep -v grep | while read line; do
            echo "📊 CPU: $(echo $line | awk '{print $3}')%"
            echo "💾 RAM: $(echo $line | awk '{print $4}')%"
            echo "⏱️  Время: $(echo $line | awk '{print $10}')"
        done
        
        echo ""
        echo "🖥️  СИСТЕМНЫЕ РЕСУРСЫ:"
        echo "  CPU Load: $(uptime | awk -F'load averages:' '{print $2}')"
        echo "  Свободной RAM: $(vm_stat | grep 'Pages free' | awk '{print int($3)*4096/1024/1024 "MB"}')"
        
        echo ""
        echo "📈 ПОСЛЕДНИЕ РЕЗУЛЬТАТЫ:"
        echo "----------------------"
        
        # Проверяем созданные файлы
        echo "📁 Созданные файлы:"
        ls -la alexnet_20_species_* 2>/dev/null | tail -5 || echo "   Пока нет файлов..."
        
        echo ""
        echo "🎯 ОЖИДАЕТСЯ ВЫСОКАЯ ТОЧНОСТЬ ДЛЯ 20 ВИДОВ!"
        
    else
        echo "🏁 ОБУЧЕНИЕ ALEXNET ЗАВЕРШЕНО!"
        
        echo ""
        echo "📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:"
        echo "======================="
        
        # Показываем созданные файлы
        echo "📁 Созданные файлы:"
        ls -la alexnet_20_species_* 2>/dev/null || echo "   Файлы не найдены"
        
        echo ""
        echo "🖼️  Визуализации:"
        ls -la alexnet_*results*.png 2>/dev/null || echo "   Графики не найдены"
        
        echo ""
        echo "📋 Последние строки отчета:"
        if ls alexnet_20_species_report_*.txt 1> /dev/null 2>&1; then
            tail -10 alexnet_20_species_report_*.txt | head -20
        else
            echo "   Отчет не найден"
        fi
        
        break
    fi
    
    echo ""
    echo "🔄 Обновление через 30 секунд... (Ctrl+C для выхода)"
    sleep 30
done

echo ""
echo "🎉 МОНИТОРИНГ ЗАВЕРШЕН!" 