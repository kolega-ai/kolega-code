## Expo React Native + Hono Backend Template

### Project Overview
This is a full stack mobile-first starter template with an Expo React Native frontend and Hono backend. The project uses TypeScript throughout with strict configuration for better code quality. The mobile app runs on port 9001 with tunneling support, the backend on port 9002, and both have been started automatically with hot reload enabled. **Note: This template prioritizes mobile app development (iOS/Android) and is not intended for web deployment.**

### Project Structure
```
/
├── frontend/          # Expo React Native mobile application
├── backend/           # Hono API server with MongoDB
```

#### Mobile Frontend Development (Port 9001)
- **Framework**: Expo SDK ~54.0.10 with React Native 0.81.4
- **React**: React 19.1.0 with latest features
- **TypeScript**: Strict configuration enabled for better code quality
- **Styling**: React Native StyleSheet API with inline styling patterns
- **Development**: Expo CLI with tunneling, device testing, and hot reload
- **Architecture**: New Architecture enabled for improved performance
- **Entry Point**: `index.ts` using `registerRootComponent(App)`
- **Main Component**: `App.tsx` with backend health checking integration
- **API Layer**: Structured in `lib/backend.ts` (business logic) and `lib/api.ts` (transport layer)

#### Backend Development (Port 9002)
- **Framework**: Hono 4.6.14 with MongoDB
- **TypeScript**: Loose configuration, builds to `./build` directory
- **CORS**: Configured to accept requests from any origin (`*`)
- **Database**: MongoDB connection via `MONGODB_URI` environment variable
- **Entry point**: `backend/index.ts`
- **Hot reload**: Enabled via tsx watch mode

### Key Conventions

#### TypeScript
- Frontend uses TypeScript ~5.9.2 with strict mode enabled
- Backend uses TypeScript with loose configuration for easier development
- ES modules with `.js` extensions in imports for backend
- Strong type safety encouraged for mobile app development

#### Code Style
- Use single quotes over double quotes
- Prefer descriptive variable names (avoid abbreviations)
- Modularize code into smaller functions
- Follow DRY principles - deduplicate code whenever possible
- Use type hints in function definitions
- Follow React Native and Expo best practices

#### File Organization
- Mobile app components go in `frontend/` root or organized subdirectories
- Backend routes should be modularized in separate files
- Keep database logic in `backend/database/`
- API integration logic in `frontend/lib/` directory
- Assets in `frontend/assets/` for app icons, splash screens, etc.

#### Making Changes

1. **Adding Mobile App Features**:
   - Create new React Native components using `View`, `Text`, `ScrollView`, `TouchableOpacity`
   - Use StyleSheet API for styling components
   - Import and use React Native components from 'react-native'
   - Add new screens and implement navigation with React Navigation (when needed)
   - Test on both iOS and Android platforms

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
   - Mobile app uses `EXPO_PUBLIC_API_BASE_URL` or `EXPO_PUBLIC_DEV_API_BASE_URL` for API endpoints
   - Environment variables for Expo must be prefixed with `EXPO_PUBLIC_`

#### Testing Changes
- Mobile app changes appear instantly via Expo hot reload
- Backend changes reload automatically via tsx
- To test on physical devices, ask user to scan QR code
- Use iOS Simulator and Android Emulator for preview builds testing
- Build production apps using EAS Build for app store deployment

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

### Expo React Native Mobile Development Cheat Sheet

#### Essential React Native Components
```typescript
import { View, Text, ScrollView, TouchableOpacity, TextInput, Alert } from 'react-native';

// Basic container and layout
<View style={styles.container}>
  <Text style={styles.title}>Hello Mobile!</Text>
</View>

// Scrollable content
<ScrollView contentContainerStyle={styles.scrollContainer}>
  <Text>Scrollable content here</Text>
</ScrollView>

// Interactive button
<TouchableOpacity style={styles.button} onPress={handlePress}>
  <Text style={styles.buttonText}>Press Me</Text>
</TouchableOpacity>

// Text input
<TextInput
  style={styles.input}
  placeholder="Enter text"
  value={inputValue}
  onChangeText={setInputValue}
/>
```

#### StyleSheet API Patterns
```typescript
import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f1f5f9',
    paddingHorizontal: 24,
  },
  centered: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3, // Android shadow
  },
  button: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: '600',
    textAlign: 'center',
  },
  responsive: {
    width: width * 0.9, // 90% of screen width
    minHeight: height * 0.6, // 60% of screen height
  },
});
```

#### Backend Integration Pattern
```typescript
// In lib/backend.ts - Business logic functions
import { apiGet, apiPost } from './api';

export async function getUserProfile(userId: string) {
  try {
    const response = await apiGet(`/api/users/${userId}`);
    const data = await response.json();
    return data.user;
  } catch (error) {
    throw new Error('Failed to fetch user profile');
  }
}

export async function createUser(userData: any) {
  try {
    const response = await apiPost('/api/users', userData);
    const data = await response.json();
    return data.user;
  } catch (error) {
    throw new Error('Failed to create user');
  }
}

// In your component
import { getUserProfile } from '../lib/backend';

export default function UserScreen() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadUserProfile();
  }, []);

  const loadUserProfile = async () => {
    try {
      setLoading(true);
      const userData = await getUserProfile('user123');
      setUser(userData);
    } catch (error) {
      Alert.alert('Error', 'Failed to load user profile');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <Text>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{user?.name}</Text>
    </View>
  );
}
```

#### Common Mobile Development Patterns
```typescript
// Loading states
const [isLoading, setIsLoading] = useState(false);

// Safe area handling (for newer devices)
import { useSafeAreaInsets } from 'react-native-safe-area-context';
const insets = useSafeAreaInsets();

// Platform-specific code
import { Platform } from 'react-native';
const isIOS = Platform.OS === 'ios';
const isAndroid = Platform.OS === 'android';

// Keyboard handling
import { KeyboardAvoidingView } from 'react-native';
<KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
  {/* Your form content */}
</KeyboardAvoidingView>

// Alert dialogs
import { Alert } from 'react-native';
Alert.alert(
  'Confirm Action',
  'Are you sure you want to continue?',
  [
    { text: 'Cancel', style: 'cancel' },
    { text: 'OK', onPress: () => console.log('Confirmed') },
  ]
);
```

#### Expo Specific Features
```typescript
// Status bar
import { StatusBar } from 'expo-status-bar';
<StatusBar style="dark" />

// Asset loading
import { Image } from 'react-native';
<Image source={require('./assets/icon.png')} style={styles.logo} />

// Environment variables
const apiUrl = process.env.EXPO_PUBLIC_API_BASE_URL;
```
