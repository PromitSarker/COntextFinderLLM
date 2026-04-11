# Stage 1: Build React App
FROM node:18-slim AS build

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy built files from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Custom Nginx config to handle React Router if needed, and optional proxying
# For now, standard Nginx config is fine as it's a simple SPA.
# If we need proxying to avoid CORS, we'd add it here.

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
