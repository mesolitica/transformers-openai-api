# Stress test

## T5 on RTX 3090 Ti

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

```bash
locust -f t5_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
locust -f t5_without_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
```

### Non-continuous batch

![alt text](t5_without_continuous.png)

### Continuous batch

![alt text](t5_continuous.png)

## Mistral 7b GPTQ on RTX 3090 Ti

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

```bash
locust -f mistral_7b_gtpq_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
locust -f mistral_7b_gtpq_without_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
```

### Non-continuous batch

![alt text](mistral_7b_gtpq_without_continuous.png)

### Continuous batch

![alt text](mistral_7b_gtpq_continuous.png)

## Whisper Large V3 on RTX 3090 Ti

Rate of 5 users per second, total requests up to 30 users for 60 seconds,

```bash
locust -f whisper_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 30 -t 60
locust -f whisper_without_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 30 -t 60
```

### Non-continuous batch

![alt text](whisper_without_continuous.png)

### Continuous batch

![alt text](whisper_continuous.png)