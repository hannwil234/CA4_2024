(define (domain robplan)
(:requirements :typing :durative-actions :strips :fluents)
  (:types  
     turtlebot robot camera camera_eo camera_ir robo_arm charger - vehicle
     vehicle photo valve pump pipe sound gas_ind obj battery - subject
     city_location city - location
     waypoint battery_station - city_location
     route
     )

  (:predicates
    (at ?physical_obj1 - subject ?location1 - location)
    (available ?vehicle1 - vehicle)
    (available ?camera1 - camera)    
    (connects ?route1 - route ?location1 - location ?location2 - location)
    (in_city ?location1 - location ?city1 - city)
    (route_available ?route1 - route)
    (no_photo ?subject1 - subject)
    (photo ?subject1 - subject)
    (no_seals_check ?subject1 - subject) ;Pump
    (seals_check ?subject1 - subject) ;Pump
    (on_robot ?Sub - subject ?V - vehicle)
    (fullBat ?B - battery)
    (noBat ?B - battery)
    (valve_manipulated ?V - valve)
    (not_valve_manipulated ?V - valve)


   )

(:functions 
           (distance ?O - location ?L - location)
           (route-length ?O - route)
	         (speed ?V - vehicle)
            )
 ;Due to battery requirement, the robot must now start at a charger to find a solution
  (:durative-action move_robot
       :parameters ( ?V - robot ?O - location ?L - location ?R - route ?B - battery)
       :duration (= ?duration (/ (route-length ?R) (speed ?V)))
       :condition (and 
			          (at start (at ?V ?O))
                (at start (available ?V))
                (at start (route_available ?R))
            		(at start (connects ?R ?O ?L))
                (at start (on_robot ?B ?V))
                ;(at start (fullBat ?B)) ;Remove comment, the robot must start at a charger
                
                
       )
       :effect (and 
		              (at start (not (at ?V ?O)))
                  (at end (at ?V ?L))
                  
                  
                  
                  
        )
    )


 (:durative-action check_seals_pump_picture_EO
       :parameters ( ?V - robot ?L - location ?G - camera_eo ?P - pump ?B - battery)
       :duration (= ?duration 5)
       :condition (and 
            (over all (at ?V ?L))
            (at start (at ?V ?L))
            (at end (at ?V ?L))
            (at start (at ?P ?L))
            (at start (available ?G))
            (at start (on_robot ?G ?V))
            (at start (no_seals_check ?P))
            (at start (on_robot ?B ?V))
            (at start (fullBat ?B))
            (at start (no_photo ?P))
       )
       :effect (and 
	    (at start (not (no_seals_check ?P)))
            (at end (seals_check ?P))
            (at end (available ?G))
            (at start (not (no_photo ?P)))
            (at end (photo ?P))
        )
    )


  (:durative-action manipulate_valve
       :parameters ( ?R - robot ?L - location ?arm - robo_arm ?V - valve ?B - battery)
       :duration (= ?duration 10)
       :condition (and 
            (over all (at ?R ?L))
            (at start (at ?R ?L))
            (at end (at ?R ?L))
            (at start (at ?V ?L))
            (at start (available ?arm))
            (at start (on_robot ?arm ?R))
            (at start (on_robot ?B ?R))
            (at start (fullBat ?B))
            (at start (not_valve_manipulated ?V))
       )
       :effect (and 
	    (at start (not (not_valve_manipulated ?V)))
            (at end (valve_manipulated ?V))
            
        )
    )
(:durative-action charge_robot
  :parameters (?R - robot ?loc - location ?C - charger ?B - battery)
  :duration(= ?duration 15)
  :condition (and
            (at start (at ?R ?loc))
            (over all (at ?R ?loc))
            (at end (at ?R ?loc))
            (at start (at ?C ?loc))
            (at start (on_robot ?B ?R))
            (at start (noBat ?B))
  )
  :effect (and 
      (at start (not (noBat ?B)))
      (at end (fullBat ?B))
  
  )
  
  
  )

  

)
