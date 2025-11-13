import lgpio
import time

# === CONFIGURACIÓN ===
CHIP = 4            # Controlador GPIO de la Raspberry Pi 5
PIN_SERVO1 = 14     # GPIO 14
PIN_SERVO2 = 15     # GPIO 15
FREQ = 50           # Frecuencia PWM de 50 Hz (20 ms)

# === FUNCIONES ===
def set_angle(chip, pin, angle):
    """Convierte un ángulo (0–180°) en un pulso PWM."""
    # 1 ms → 0°, 2 ms → 180°
    pulse = 0.0005 + (angle / 180.0) * 0.002
    duty = (pulse / 0.02) * 100
    lgpio.tx_pwm(chip, pin, FREQ, duty)
    time.sleep(0.3)       # Dar tiempo al servo para moverse
    lgpio.tx_pwm(chip, pin, 0, 0)  # Apagar PWM para evitar vibraciones

# === INICIALIZACIÓN ===
h = lgpio.gpiochip_open(CHIP)
lgpio.gpio_claim_output(h, PIN_SERVO1)
lgpio.gpio_claim_output(h, PIN_SERVO2)

print("Control de servos iniciado.")
print("Introduce dos ángulos separados por espacio (ejemplo: 90 45)")
print("Pulsa Ctrl+C para salir.\n")

try:
    while True:
        # Leer entrada del usuario
        entrada = input("Ángulos (servo1 servo2): ").strip()
        if not entrada:
            continue

        try:
            ang1_str, ang2_str = entrada.split()
            ang1 = max(0, min(180, int(ang1_str)))
            ang2 = max(0, min(180, int(ang2_str)))
        except ValueError:
            print("⚠️ Ingresa dos valores válidos, por ejemplo: 90 45")
            continue

        # Mover los servos
        set_angle(h, PIN_SERVO1, ang1)
        set_angle(h, PIN_SERVO2, ang2)

        print(f"Servo1 → {ang1}° | Servo2 → {ang2}°")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nSaliendo...")

finally:
    # Detener PWM y liberar GPIO
    lgpio.tx_pwm(h, PIN_SERVO1, 0, 0)
    lgpio.tx_pwm(h, PIN_SERVO2, 0, 0)
    lgpio.gpiochip_close(h)
    print("GPIO liberados.")