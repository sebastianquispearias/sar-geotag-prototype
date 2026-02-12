"""
Algoritmos de geolocalización del paper.

Implementación de "Object Geolocalization Using Consumer-Grade Devices"
Sección 4: Proposed Geolocalization Algorithms

Algoritmos implementados:
- Ray Intersection (Sección 4.2): Para 2 mediciones - triangulación
- Gradient Descent (Sección 4.3): Para 3+ mediciones - optimización

El algoritmo Ray Marching (Sección 4.1) requiere un DEM (Digital Elevation Model)
que no está disponible en este prototipo.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.geometry.ecef import ECEFPoint, WGS84Point, ecef_to_wgs84, distance_ecef
from src.geometry.pinhole import Measurement


def ray_intersection(m1: Measurement, m2: Measurement) -> Tuple[Optional[ECEFPoint], float]:
    """
    Algoritmo Ray Intersection del paper (Sección 4.2).
    
    Calcula la geolocalización de un objeto usando 2 mediciones desde
    posiciones diferentes (triangulación).
    
    Del paper:
    "For cases where an object is not on ground level, the geolocation can be 
    estimated by two measurements from different positions, i.e., triangulation...
    
    A challenge in our setting in three-dimensional space is that two rays are 
    extremely unlikely to intersect; two rays are typically skew to each other. 
    Intuitively, the point in question would be in the middle, where both rays 
    are nearest."
    
    Método:
    1. Calcular u = v1 × v2 (ortogonal a ambos rayos)
    2. Resolver el sistema lineal:
       α·v1 + β·u - γ·v2 = p2 - p1
    3. El punto estimado es:
       q = p1 + α·v1 + (β·u)/2
    
    Parámetros
    ----------
    m1 : Measurement
        Primera medición (origin p1, direction v1)
    m2 : Measurement
        Segunda medición (origin p2, direction v2)
    
    Retorna
    -------
    (point, distance) : Tuple[Optional[ECEFPoint], float]
        - point: Punto estimado en ECEF, o None si los rayos son paralelos
        - distance: Distancia mínima entre los dos rayos (indica calidad)
    """
    p1 = m1.origin.to_array()
    v1 = m1.direction
    p2 = m2.origin.to_array()
    v2 = m2.direction

    # Verificar que los rayos no sean paralelos
    u = np.cross(v1, v2)
    u_norm = np.linalg.norm(u)

    if u_norm < 1e-10:
        # Rayos paralelos, no se puede triangular
        return None, float("inf")

    u = u / u_norm  # Normalizar

    # Construir el sistema lineal: [v1 | u | -v2] @ [α, β, γ]^T = p2 - p1
    # Matriz de coeficientes (3x3)
    A = np.column_stack([v1, u, -v2])
    b = p2 - p1

    # Resolver usando Gauss-Jordan (o np.linalg.solve)
    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, float("inf")

    alpha, beta, gamma = solution

    # Punto estimado: q = p1 + α·v1 + (β·u)/2
    # (El punto medio de la línea ortogonal a ambos rayos)
    q = p1 + alpha * v1 + (beta / 2.0) * u

    # La distancia mínima entre los rayos es |β| (longitud de la línea ortogonal)
    ray_distance = abs(beta)

    return ECEFPoint.from_array(q), ray_distance


def gradient_descent(
    measurements: List[Measurement],
    learning_rate: float = 0.065,
    max_iterations: int = 1000,
    tolerance: float = 1e-7,
    initial_guess: Optional[ECEFPoint] = None,
) -> Tuple[Optional[ECEFPoint], float]:
    """
    Algoritmo Gradient Descent del paper (Sección 4.3).
    
    Encuentra la geolocalización óptima para 3 o más mediciones minimizando
    la suma de distancias ortogonales al cuadrado.
    
    Del paper:
    "By building upon the results of Traa [21] we can rephrase geolocalization 
    as a minimization problem for a set of measurements, numerically solvable 
    by gradient descent."
    
    Función de pérdida (Traa [21, Eq. 10]):
        loss(M; q) = Σ d(pi, vi; q)
        donde d(p, v; q) = (p - q)^T · (I - v·v^T) · (p - q)
    
    Derivada parcial (Traa [21, Eq. 12]):
        ∂loss/∂q = -2 · Σ (I - vi·vi^T) · (pi - q)
    
    Parámetros
    ----------
    measurements : List[Measurement]
        Conjunto de mediciones (mínimo 2, idealmente 3+)
    learning_rate : float
        Tasa de aprendizaje (paper usa 0.065)
    max_iterations : int
        Máximo de iteraciones
    tolerance : float
        Criterio de convergencia (paper usa 1e-7)
    initial_guess : Optional[ECEFPoint]
        Estimación inicial (si None, usa promedio de orígenes)
    
    Retorna
    -------
    (point, loss) : Tuple[Optional[ECEFPoint], float]
        - point: Punto estimado en ECEF
        - loss: Valor final de la función de pérdida
    """
    if len(measurements) < 2:
        return None, float("inf")

    # Pre-computar las matrices (I - v·v^T) para cada medición
    # Esto es la optimización mencionada en el paper
    projection_matrices = []
    origins = []
    for m in measurements:
        v = m.direction.reshape(3, 1)
        I_minus_vvT = np.eye(3) - v @ v.T
        projection_matrices.append(I_minus_vvT)
        origins.append(m.origin.to_array())

    # Estimación inicial
    if initial_guess is not None:
        q = initial_guess.to_array()
    else:
        # Promedio de los orígenes como punto inicial
        q = np.mean(origins, axis=0)

    # Gradient descent loop (Algorithm 2 del paper)
    q_prev = np.zeros(3)

    for iteration in range(max_iterations):
        # Calcular derivada parcial
        dp = np.zeros(3)
        for i, (P, p) in enumerate(zip(projection_matrices, origins)):
            dp -= 2.0 * P @ (p - q)

        # Actualizar q
        q = q - learning_rate * dp

        # Verificar convergencia
        v = q - q_prev
        if np.linalg.norm(v) < tolerance:
            break

        q_prev = q.copy()

    # Calcular loss final
    loss = 0.0
    for P, p in zip(projection_matrices, origins):
        diff = p - q
        loss += diff.T @ P @ diff

    return ECEFPoint.from_array(q), loss


def geolocalize_from_measurements(
    measurements: List[Measurement],
) -> Tuple[Optional[WGS84Point], dict]:
    """
    Geolocaliza un objeto usando el algoritmo apropiado según el número de mediciones.
    
    Del paper (Figure 3 - Algorithm selection process):
    - 1 medición: Ray Marching (requiere DEM, no implementado)
    - 2 mediciones: Ray Intersection
    - 3+ mediciones: Gradient Descent
    
    Parámetros
    ----------
    measurements : List[Measurement]
        Lista de mediciones del mismo objeto
    
    Retorna
    -------
    (point, info) : Tuple[Optional[WGS84Point], dict]
        - point: Geolocalización en WGS84, o None si falla
        - info: Diccionario con información adicional (algoritmo usado, distancia, etc.)
    """
    n = len(measurements)
    info = {"algorithm": None, "num_measurements": n}

    if n < 2:
        info["error"] = "Se necesitan al menos 2 mediciones para triangular"
        return None, info

    if n == 2:
        # Ray Intersection
        point_ecef, ray_dist = ray_intersection(measurements[0], measurements[1])
        info["algorithm"] = "ray_intersection"
        info["ray_distance_m"] = ray_dist

        if point_ecef is None:
            info["error"] = "Rayos paralelos, no se puede triangular"
            return None, info

        # Calcular distancia desde la primera cámara
        info["distance_from_cam1_m"] = distance_ecef(measurements[0].origin, point_ecef)
        info["distance_from_cam2_m"] = distance_ecef(measurements[1].origin, point_ecef)

        point_wgs84 = ecef_to_wgs84(point_ecef)
        return point_wgs84, info

    else:
        # Gradient Descent (3+ mediciones)
        # Usar ray_intersection de las primeras 2 como estimación inicial
        initial, _ = ray_intersection(measurements[0], measurements[1])

        point_ecef, loss = gradient_descent(
            measurements,
            initial_guess=initial,
        )
        info["algorithm"] = "gradient_descent"
        info["loss"] = loss

        if point_ecef is None:
            info["error"] = "Gradient descent no convergió"
            return None, info

        info["distance_from_cam1_m"] = distance_ecef(measurements[0].origin, point_ecef)

        point_wgs84 = ecef_to_wgs84(point_ecef)
        return point_wgs84, info


def validate_measurements(m1: Measurement, m2: Measurement) -> dict:
    """
    Valida si dos mediciones son adecuadas para triangulación.
    
    Del paper:
    "Adding additional measurements is only considered when the origin is unique 
    and v is not parallel to existing measurements."
    
    Retorna un diccionario con información sobre la validez.
    """
    result = {
        "valid": True,
        "warnings": [],
    }

    # Verificar que los orígenes sean diferentes
    origin_dist = distance_ecef(m1.origin, m2.origin)
    if origin_dist < 0.5:  # Menos de 0.5m de separación
        result["valid"] = False
        result["warnings"].append(
            f"Los orígenes están muy cerca ({origin_dist:.2f}m). "
            "Muévete al menos 1-2 metros entre mediciones."
        )

    # Verificar que los rayos no sean paralelos
    dot_product = abs(np.dot(m1.direction, m2.direction))
    angle_deg = math.degrees(math.acos(min(dot_product, 1.0)))

    if angle_deg < 5:
        result["valid"] = False
        result["warnings"].append(
            f"Los rayos son casi paralelos ({angle_deg:.1f}°). "
            "Muévete más lateralmente para mejor triangulación."
        )
    elif angle_deg < 15:
        result["warnings"].append(
            f"Ángulo entre rayos bajo ({angle_deg:.1f}°). "
            "Mayor separación lateral mejoraría la precisión."
        )

    result["origin_separation_m"] = origin_dist
    result["ray_angle_deg"] = angle_deg

    return result
