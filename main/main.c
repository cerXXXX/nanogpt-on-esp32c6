#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "inference.h"
#include "esp_timer.h"

// Настройки WiFi
#define WIFI_SSID      "Neige"
#define WIFI_PASS      "iv09ma14al16an19"

static const char *TAG = "MAIN";
Transformer transformer;

/* HTML страница */
const char* html_page = 
"<!DOCTYPE html><html><head><title>ESP32-C6 LLM</title>"
"<meta name='viewport' content='width=device-width, initial-scale=1'>"
"<link rel='icon' href='data:,'>" // Отключаем запрос фавиконки в самом HTML на всякий случай
"</head>"
"<body><h1>ESP32-C6 NanoLLM</h1>"
"<form action='/generate' method='POST'><input type='submit' value='Generate Story'></form>"
"<div id='out' style='white-space: pre-wrap; font-family: monospace; margin-top:20px'></div>"
"<script>"
"const form = document.querySelector('form');"
"form.addEventListener('submit', async (e) => {"
"  e.preventDefault();"
"  const out = document.getElementById('out'); out.innerText = '';"
"  const response = await fetch('/generate', {method:'POST'});"
"  const reader = response.body.getReader();"
"  const decoder = new TextDecoder();"
"  while (true) {"
"    const { done, value } = await reader.read();"
"    if (done) break;"
"    out.innerText += decoder.decode(value);"
"  }"
"});"
"</script></body></html>";

esp_err_t generate_post_handler(httpd_req_t *req) {
    // Увеличим лимит слов, чтобы история была подлиннее
    const int max_tokens = 100; 
    
    int token = 1; // Начальный токен (BOS - Beginning of Sentence)
    int pos = 0;
    
    // Говорим браузеру: "Это текст, соединение закрывай сам"
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_set_hdr(req, "Transfer-Encoding", "chunked"); // Явный chunked
    
    char buffer[256];
    int64_t start_time = esp_timer_get_time();

    for (pos = 0; pos < max_tokens; pos++) {
        // 1. Вычисляем следующий токен
        float* logits = forward(&transformer, token, pos);
        int next_token = sample(&transformer, logits);
        
        // 2. Проверка на конец текста (EOS)
        // У Llama 2 токен 2 обычно значит "Конец текста"
        if (next_token == 2 || next_token == 0) {
            ESP_LOGI(TAG, "End of story reached.");
            break; 
        }

        // 3. Декодируем токен в строку
        char* piece = decode(&transformer, token, next_token);
        
        // 4. Отправляем, если строка валидная
        if (piece && strlen(piece) > 0) {
            // Для отладки пишем в консоль, чтобы видеть прогресс
            printf("%s", piece); 
            fflush(stdout);

            // Отправляем в браузер
            httpd_resp_send_chunk(req, piece, strlen(piece));
        }

        // Сдвигаем окно
        token = next_token;
        
        // 5. Критически важно для ESP32-C6!
        // Даем процессору "подышать", чтобы не сработал Watchdog
        // и чтобы WiFi успел отправить пакет.
        vTaskDelay(1); 
    }
    
    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "\nGeneration done. Time: %lld ms", (end_time - start_time) / 1000);

    // Завершаем передачу пустым чанком
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

/* Главная страница */
esp_err_t root_get_handler(httpd_req_t *req) {
    ESP_LOGI(TAG, "Browser requested HOME page");
    
    // ВАЖНО: Говорим браузеру не держать соединение открытым
    httpd_resp_set_hdr(req, "Connection", "close");
    
    httpd_resp_send(req, html_page, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/* Заглушка для иконки */
esp_err_t favicon_get_handler(httpd_req_t *req) {
    httpd_resp_set_hdr(req, "Connection", "close"); // Тоже закрываем сразу
    httpd_resp_set_type(req, "image/x-icon");
    httpd_resp_send(req, NULL, 0);
    return ESP_OK;
}

/* Запуск сервера */
void start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    
    config.stack_size = 24576;      
    config.server_port = 80;

    // --- АГРЕССИВНЫЕ НАСТРОЙКИ СЕТИ ---
    // Увеличиваем лимиты, чтобы выдержать напор Chrome
    config.max_open_sockets = 7;     // Максимум для LWIP по умолчанию
    config.backlog_conn = 8;         // Очередь TCP соединений
    config.lru_purge_enable = true;  // Убивать старые соединения
    config.recv_wait_timeout = 5;    // Быстрый тайм-аут (5 сек)
    config.send_wait_timeout = 5;    
    
    // Глобальный контекст (опционально, но полезно для отладки)
    config.global_user_ctx = NULL;
    config.global_user_ctx_free_fn = NULL;
    config.open_fn = NULL;
    config.close_fn = NULL;

    httpd_handle_t server = NULL;
    ESP_LOGI(TAG, "Starting HTTP Server with Aggressive Config...");

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_root = { .uri = "/", .method = HTTP_GET, .handler = root_get_handler };
        httpd_register_uri_handler(server, &uri_root);
        
        httpd_uri_t uri_gen = { .uri = "/generate", .method = HTTP_POST, .handler = generate_post_handler };
        httpd_register_uri_handler(server, &uri_gen);

        httpd_uri_t uri_icon = { .uri = "/favicon.ico", .method = HTTP_GET, .handler = favicon_get_handler };
        httpd_register_uri_handler(server, &uri_icon);
        
        ESP_LOGI(TAG, "Server listening on http://192.168.0.128");
    } else {
        ESP_LOGE(TAG, "Error starting server!");
    }
}

/* WiFi Init */
static void wifi_init_sta(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    
    wifi_config_t wifi_config = { 
        .sta = { 
            .ssid = WIFI_SSID, 
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        } 
    };
    
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    esp_wifi_connect();

    // Отключаем энергосбережение!
    esp_wifi_set_ps(WIFI_PS_NONE);
    ESP_LOGI(TAG, "WiFi Power Save disabled");
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting NanoLLM on ESP32-C6...");
    
    // 1. Инициализация WiFi
    wifi_init_sta();
    
    // 2. Подготовка модели
    build_transformer(&transformer);
    
    // 3. Старт сервера
    // Небольшая задержка, чтобы WiFi успел поднять IP
    vTaskDelay(pdMS_TO_TICKS(3000)); 
    start_webserver();
    
    ESP_LOGI(TAG, "System Ready. Entering loop...");
    
    // Вечный цикл, чтобы main не завершался
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}