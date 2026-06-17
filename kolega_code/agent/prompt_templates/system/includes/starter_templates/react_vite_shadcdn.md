## React Vite Shadcn/UI Template Project

### Project Overview
This is a full stack starter template with a React frontend and Hono backend. The project uses TypeScript throughout with loose configuration for easier development. The frontend runs on port 9001, the backend on port 9002, and both have been started automatically and will restart if the code changes.

### Project Structure
```
/
├── frontend/          # React + Vite application
├── backend/           # Hono API server with MongoDB
```

#### Frontend Development (Port 9001)
- **Framework**: React 19.1.0 with Vite 7.0.4
- **Styling**: Tailwind CSS with ShadCN/UI components
- **TypeScript**: Loose configuration (no strict mode)
- **Components**: All 46 ShadCN/UI components are pre-installed in `frontend/src/components/ui/`
- **Routing**: React Router DOM with pages in `frontend/src/pages/`
- **Import paths**: Use `@/` alias for src directory (e.g., `@/components/ui/button`)

#### Backend Development (Port 9002)
- **Framework**: Hono 4.6.14 with MongoDB
- **TypeScript**: Loose configuration, builds to `./build` directory
- **CORS**: Configured to accept requests from any origin (`*`)
- **Database**: MongoDB connection via `MONGODB_URI` environment variable
- **Entry point**: `backend/index.ts`
- **Hot reload**: Enabled via tsx watch mode

### Key Conventions

#### TypeScript
- Both frontend and backend use TypeScript 5.8.3
- Loose configuration (strict: false) for easier development
- ES modules with `.js` extensions in imports for backend

#### Code Style
- Use single quotes over double quotes
- Prefer descriptive variable names (avoid abbreviations)
- Modularize code into smaller functions
- Follow DRY principles - deduplicate code whenever possible
- Use type hints in function definitions

#### File Organization
- Frontend components go in `frontend/src/components/`
- Backend routes should be modularized in separate files
- Keep database logic in `backend/database/`

#### Making Changes

1. **Adding Frontend Features**:
   - Use existing ShadCN/UI components from `@/components/ui/`
   - Add new pages in `frontend/src/pages/`
   - Update routing in `frontend/src/App.tsx`

2. **Adding Backend Endpoints**:
   - Create route modules that export Hono routers
   - Use the MongoDB connection from `database/connection.ts`
   - Maintain consistent error handling and response formats
   - The path portion of the URL should ALWAYS start with /api/

3. **Database Operations**:
   - Use `get_database()` to access the MongoDB instance
   - Handle connection errors gracefully
   - Keep connection logic centralized

4. **Environment Variables**:
   - Backend uses `MONGODB_URI` and `DATABASE_NAME`

#### Testing Changes
- Frontend changes appear instantly via HMR
- Backend changes reload automatically via tsx
- Build both projects using the `build_backend` and `build_frontend` tools to verify production readiness

### Hono Backend Cheat Sheet

#### Basic Route Handlers
```typescript
// GET request
app.get('/users', async (c) => {
  const users = await get_users_from_db();
  return c.json({ users });
});

// POST request with body parsing
app.post('/users', async (c) => {
  const body = await c.req.json();
  const { name, email } = body;
  // Process data...
  return c.json({ message: 'User created' }, 201);
});

// Route parameters
app.get('/users/:id', async (c) => {
  const id = c.req.param('id');
  const user = await get_user_by_id(id);
  return c.json({ user });
});

// Query parameters
app.get('/search', async (c) => {
  const query = c.req.query('q');
  const limit = c.req.query('limit') || '10';
  // Search logic...
  return c.json({ results });
});
```

#### MongoDB Integration Pattern
```typescript
import { get_database } from './database/connection.js';

app.get('/items', async (c) => {
  try {
    const db = get_database();
    const items = await db.collection('items').find({}).toArray();
    return c.json({ success: true, items });
  } catch (error) {
    return c.json({ success: false, error: error.message }, 500);
  }
});
```

#### Creating Modular Routes
```typescript
// routes/user_routes.ts
import { Hono } from 'hono';

export const create_user_routes = (): Hono => {
  const router = new Hono();
  
  router.get('/', async (c) => {
    // List users
  });
  
  router.post('/', async (c) => {
    // Create user
  });
  
  return router;
};

// In index.ts
import { create_user_routes } from './routes/user_routes.js';
app.route('/api/users', create_user_routes());
```

#### Common Response Patterns
```typescript
// Success response
return c.json({ success: true, data: result });

// Error response
return c.json({ success: false, error: 'Not found' }, 404);

// Redirect
return c.redirect('/new-location');

// Set headers
c.header('X-Custom-Header', 'value');
return c.json({ data });

// Set status
return c.text('Created', 201);
```

#### Middleware Pattern
```typescript
// Custom middleware
app.use('*', async (c, next) => {
  console.log(`${c.req.method} ${c.req.path}`);
  await next();
});

// Protected routes
app.use('/api/*', async (c, next) => {
  const token = c.req.header('Authorization');
  if (!token) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  await next();
});
```
