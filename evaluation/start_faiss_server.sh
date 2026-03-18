#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/faiss_server.py"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

START_PORT=9000
END_PORT=9549

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FAISS IVFPQ Cluster (550 instances)${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Ports:          9000-9549${NC}"
echo -e "${BLUE}Memory/inst:    ~1.5 GB${NC}"
echo -e "${BLUE}Total memory:   ~1.2 TB${NC}"
echo -e "${BLUE}Cluster QPS:    ~24,000${NC}"
echo -e "${BLUE}Max concurrent: 640,000${NC}"
echo -e "${BLUE}Target load:    10,000 concurrent${NC}"
echo -e "${GREEN}========================================${NC}"

if [ ! -f "$SERVER_SCRIPT" ]; then
    echo -e "${RED}Error: Server script not found${NC}"
    exit 1
fi

IVFPQ_INDEX="/ramdata/faiss_ivfpq/merged_ivfpq_from_flat.faiss"
if [ ! -f "$IVFPQ_INDEX" ]; then
    echo -e "${RED}Error: IVFPQ index not found${NC}"
    exit 1
fi

AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
REQUIRED_MEM=1300

if [ "$AVAILABLE_MEM" -lt "$REQUIRED_MEM" ]; then
    echo -e "${YELLOW}Warning: Available ${AVAILABLE_MEM}GB < Required ${REQUIRED_MEM}GB${NC}"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${YELLOW}Stopping existing instances...${NC}"
pkill -9 -f "faiss_server.py" 2>/dev/null
sleep 2

echo -e "${YELLOW}Starting 550 instances...${NC}"

BATCH_SIZE=40
BATCH_DELAY=5

total_batches=$(( (550 + BATCH_SIZE - 1) / BATCH_SIZE ))

for port in $(seq $START_PORT $END_PORT); do
    nohup python "$SERVER_SCRIPT" $port > "$LOG_DIR/faiss_$port.log" 2>&1 &

    if [ $(( ($port - $START_PORT + 1) % $BATCH_SIZE )) -eq 0 ]; then
        current_batch=$(( ($port - $START_PORT + 1) / BATCH_SIZE ))
        echo -e "  ${GREEN}✓${NC} Batch $current_batch/$total_batches: Ports $((port - BATCH_SIZE + 1))-$port"

        if [ $port -lt $END_PORT ]; then
            sleep $BATCH_DELAY
        fi
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Waiting for initialization...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Estimated: 3-5 minutes${NC}"

sleep 180

echo ""
echo -e "${YELLOW}Health Check (sampling 40 instances):${NC}"

healthy_count=0

for i in $(seq 0 39); do
    port=$(( START_PORT + i * 550 / 40 ))
    response=$(curl -s -o /dev/null -w "%{http_code}" <REDACTED_URL> 2>/dev/null)

    if [ "$response" = "200" ]; then
        healthy_count=$((healthy_count + 1))
        if [ $i -lt 5 ] || [ $i -gt 34 ]; then
            echo -e "  Port $port: ${GREEN}✓${NC}"
        elif [ $i -eq 5 ]; then
            echo "  ..."
        fi
    else
        if [ $i -lt 5 ] || [ $i -gt 34 ]; then
            echo -e "  Port $port: ${RED}✗${NC}"
        fi
    fi
done

estimated_healthy=$(( healthy_count * 550 / 40 ))
health_rate=$(( healthy_count * 100 / 40 ))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cluster Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Sampled:   $healthy_count / 40${NC}"
echo -e "${BLUE}Estimated: ~$estimated_healthy / 550 healthy${NC}"
echo -e "${BLUE}Health:    ~${health_rate}%${NC}"
echo ""
echo "Commands:"
echo "  Test:   curl <REDACTED_URL>"
echo "  Logs:   tail -f $LOG_DIR/faiss_9000.log"
echo "  Stop:   ./stop_faiss_cluster.sh"
echo ""
