import React from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Home,
  Upload,
  RefreshCw,
  BarChart3,
  Activity,
  Stethoscope,
} from "lucide-react";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const navigation = [
    { name: "Dashboard", href: "/", icon: Home },
    { name: "Predict", href: "/predict", icon: Upload },
    { name: "Retrain", href: "/retrain", icon: RefreshCw },
    { name: "Visualizations", href: "/visualizations", icon: BarChart3 },
  ];

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="flex flex-col w-64 bg-white shadow-lg">
        {/* Logo */}
        <div className="flex items-center px-6 py-4 bg-primary-600">
          <Stethoscope className="h-8 w-8 text-white mr-3" />
          <h1 className="text-xl font-bold text-white">X-Ray Classifier</h1>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                  isActive(item.href)
                    ? "bg-primary-100 text-primary-700 border-r-2 border-primary-500"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
              >
                <Icon
                  className={`mr-3 h-5 w-5 ${
                    isActive(item.href) ? "text-primary-500" : "text-gray-400"
                  }`}
                />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-gray-200">
          <div className="flex items-center">
            <Activity className="h-5 w-5 text-green-500 mr-2" />
            <span className="text-sm text-gray-600">System Active</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
          <div className="container mx-auto px-6 py-8">{children}</div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
