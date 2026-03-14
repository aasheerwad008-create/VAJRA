FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run compile

EXPOSE 8545

CMD ["npx", "hardhat", "node"]
