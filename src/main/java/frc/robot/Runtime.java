package frc.robot;

import frc.robot.commands.*;
import frc.robot.modules.common.*;
import frc.robot.modules.common.Input.*;
import frc.robot.modules.common.drive.DriveBase;
import frc.robot.modules.common.drive.*;
import frc.robot.modules.common.EventTriggers.*;
import frc.robot.modules.vision.java.*;

import edu.wpi.first.wpilibj.DriverStation;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj2.command.CommandScheduler;
import edu.wpi.first.wpilibj2.command.ParallelCommandGroup;
import edu.wpi.first.wpilibj2.command.SequentialCommandGroup;
import edu.wpi.first.wpilibj2.command.button.Trigger;
import edu.wpi.first.networktables.NetworkTableInstance;


//comment

/* TODO:
 x Split VisionServer into base and an extension that integrates command-based structure (this would extend SubsystemBase)
 x Verify/imporve singleton for each ^^^
 x Create "helper" container for methods at the start of DriveBase
 x Make better names for "DB#..." and split into port map and motorcontroller object containers (this would be an extension)
 - Commands for CLDifferential -> Ramsete controller (characterization first)
 - Update C++ VisionServer-Robot API (actually already safe because of the use of LANGUAGE SUPPORT FOR UNSIGNED ~> smh java :| )
 - Methods/impelemtation to search DriverStation for a certain input (common.Input.InputDevice) and return object/port
 x max output/scaling method for drivebase
 - make a spreadsheet for camera presets (each pipeline) under different lighting conditions
 x? fix controls being f'ed when not in sim mode and not connected on program startup
 - finalize/test Velocity-CL (TalonFX) shooter commands
 x? Polish cargo manipulation controls
 - AUTO!!!!
 - make AnalogSupplier and DigitalSupplier extend BooleanSupplier and DigitalSupplier respectively

 - Tune hubturn p-loop
 - Change camera params / configure switching when camera positions are finalized
*/

public class Runtime extends TimedRobot {

	private final InputDevice
		input = new InputDevice(0),			// xbox controller

		stick_left = new InputDevice(1),	// acrade stick (left)
		stick_right = new InputDevice(2);	// arcade stick (right)

	private final DriveBase drivebase = new DriveBase(Constants.drivebase_map_2022);
	private final CargoSystemV2 cargo_sys = new CargoSystemV2(
		new CargoSystemV2.IntakeSubsystem(Constants.intake_port),
		new CargoSystemV2.TransferSubsystem(Constants.transfer_ports),
		new CargoSystemV2.ShooterSubsystem(Motors.pwm_victorspx, Constants.w0_shooter_port, Motors.pwm_victorspx, Constants.feed_port)
	);

	//private boolean has_bindings = false;


	public Runtime() {
		System.out.println("RUNTIME INITIALIZATION");

		this.drivebase.setSpeedScaling(Constants.teleop_drivebase_scaling);
		this.drivebase.setSpeedDeadband(Constants.teleop_drivebase_deadband);
		this.drivebase.setSpeedSquaring(Constants.teleop_drivebase_speed_squaring);

		this.cargo_sys.startAutomaticTransfer(Constants.transfer_speed);
	}

	@Override public void robotPeriodic() { CommandScheduler.getInstance().run(); }
	@Override public void robotInit() {
		//AutonomousTrigger.Get().whenActive( new Auto.WeekZero(this.drivebase, this.w0_cargo_sys) );
		//TestTrigger.Get().whenActive( new CargoFollow.Demo(this.drivebase, DriverStation.getAlliance(), Constants.cargo_cam_name) );

		new Trigger(()->VisionServer.isConnected()).whenActive(new LambdaCommand(()->System.out.println("VisionServer Connected")));

		// if(this.input.isConnected()) {
		// 	this.xboxControls();
		// } else {
		// 	this.input.connectionTrigger().whenActive(
		// 		new LambdaCommand.Singular(()->{
		// 			this.xboxControls();
		// 			System.out.println("Xbox Bindings Scheduled.");
		// 		}, true)
		// 	);
		// }
		if(this.stick_left.isConnected() && this.stick_right.isConnected()) {
			this.arcadeControls();
		} else {
			this.stick_left.connectionTrigger().and(this.stick_right.connectionTrigger()).whenActive(
				new LambdaCommand.Singular(()->{
					this.arcadeControls();
					System.out.println("Arcade Bindings Scheduled.");
				}, true)
			);
		}

		// this.input.connectionTrigger().and(
		// 	this.stick_left.connectionTrigger().and(this.stick_right.connectionTrigger()).negate()
		// ).and(
		// 	new Trigger(()->this.has_bindings).negate()
		// ).whenActive(
		// 	new LambdaCommand.Singular(()->{
		// 		this.xboxControls();
		// 		System.out.println("Xbox Bindings Scheduled.");
		// 		this.has_bindings = true;
		// 	}, true)
		// );
		// this.stick_left.connectionTrigger().and(this.stick_right.connectionTrigger()).and(
		// 	this.input.connectionTrigger().negate()
		// ).and(
		// 	new Trigger(()->this.has_bindings).negate()
		// ).whenActive(
		// 	new LambdaCommand.Singular(()->{
		// 		this.arcadeControls();
		// 		System.out.println("Arcade Bindings Scheduled.");
		// 		this.has_bindings = true;
		// 	}, true)
		// );

	}

	@Override public void disabledInit() {}
	@Override public void disabledPeriodic() {}
	@Override public void disabledExit() {}

	@Override public void autonomousInit() {}
	@Override public void autonomousPeriodic() {}
	@Override public void autonomousExit() {}

	@Override public void teleopInit() {}
	@Override public void teleopPeriodic() {}
	@Override public void teleopExit() {}

	@Override public void testInit() {}
	@Override public void testPeriodic() {}
	@Override public void testExit() {}



	private void xboxControls() {	// bindings for xbox controller

		TeleopTrigger.Get().whenActive(
			new SequentialCommandGroup(
				new LambdaCommand(()->VisionServer.setStatistics(false)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(false)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_driving))
			)
		).whenActive(
			this.drivebase.modeDrive(
				Xbox.Analog.LX.getSupplier(input),
				Xbox.Analog.LY.getSupplier(input),
				Xbox.Analog.LT.getSupplier(input),
				Xbox.Analog.RX.getSupplier(input),
				Xbox.Analog.RY.getSupplier(input),
				Xbox.Analog.RT.getSupplier(input),
				Xbox.Digital.RS.getPressedSupplier(input),
				Xbox.Digital.LS.getPressedSupplier(input)
			), false
		);	// schedule mode drive when in teleop mode

		Xbox.Digital.DT.getCallbackFrom(this.input).whenPressed(VisionSubsystem.IncrementPipeline.Get());	// dpad top -> increment pipeline
		Xbox.Digital.DB.getCallbackFrom(this.input).whenPressed(VisionSubsystem.DecrementPipeline.Get());	// dpad bottom -> decrement pipeline
		Xbox.Digital.DR.getCallbackFrom(this.input).whenPressed(VisionSubsystem.IncrementCamera.Get());		// dpad right -> increment camera
		Xbox.Digital.DL.getCallbackFrom(this.input).whenPressed(VisionSubsystem.DecrementCamera.Get());		// dpad left -> decrement camera
		Xbox.Digital.START.getCallbackFrom(this.input).whenPressed(VisionSubsystem.ToggleProcessing.Get());	// toggle visionserver processing
		Xbox.Digital.BACK.getCallbackFrom(this.input).whenPressed(VisionSubsystem.ToggleStatistics.Get());	// toggle statistics in camera view

		Xbox.Digital.X.getCallbackFrom(this.input).and(				// when 'X' is pressed...
			Xbox.Digital.LB.getCallbackFrom(this.input).negate()	// and 'LB' IS NOT pressed...
		).and( TeleopTrigger.Get() ).whileActiveOnce(				// and in teleop mode...
			this.cargo_sys.managedIntake(Constants.intake_speed)	// run the intake (managed)
		);
		Xbox.Digital.X.getCallbackFrom(this.input).and(				// when 'X' is pressed...
			Xbox.Digital.LB.getCallbackFrom(this.input)				// and 'LB' IS pressed...
		).and( TeleopTrigger.Get() ).whileActiveOnce(				// and in teleop mode...
			this.cargo_sys.basicIntake(Constants.intake_speed)		// override the intake (unmanaged)
		);

		Xbox.Digital.Y.getCallbackFrom(this.input).and(				// when 'Y' is pressed...
			Xbox.Digital.LB.getCallbackFrom(this.input)				// and 'LB' IS pressed...
		).and( TeleopTrigger.Get() ).whileActiveOnce(				// and in teleop mode...
			this.cargo_sys.basicTransfer(Constants.transfer_speed)	// override the transfer belts (unmanaged)
		);
		Xbox.Digital.B.getCallbackFrom(this.input).and(				// when 'B' is pressed...
			Xbox.Digital.LB.getCallbackFrom(this.input).negate()	// and 'LB' IS NOT pressed...
		).and(
			Xbox.Digital.RB.getToggleFrom(this.input).negate()		// and 'RB' IS NOT toggled...
		).and( TeleopTrigger.Get() ).toggleWhenActive(				// and in teleop mode...
			this.cargo_sys.managedShoot(							// control the shooter (managed)
				()->Xbox.Digital.A.getValueOf(this.input),
				Constants.feed_speed,
				Constants.shooter_default_speed
			)
		);
		Xbox.Digital.B.getCallbackFrom(this.input).and(				// when 'B' is pressed...
			Xbox.Digital.LB.getCallbackFrom(this.input)				// and 'LB' IS pressed...
		).and(
			Xbox.Digital.RB.getToggleFrom(this.input).negate()		// and 'RB IS NOT toggled...'
		).and( TeleopTrigger.Get() ).toggleWhenActive(				// and in teleop mode...
			this.cargo_sys.basicShoot(								// control the shooter (unmanaged)
				()->Xbox.Digital.A.getValueOf(this.input),
				Constants.feed_speed,
				Constants.shooter_default_speed
			)
		);

		Xbox.Digital.RB.getToggleFrom(this.input).and(
			TeleopTrigger.Get()
		).whenActive(
			new SequentialCommandGroup(
				new LambdaCommand(()->System.out.println("VISION ASSIST RUNNING...")),
				new LambdaCommand(()->this.drivebase.modeDrive().cancel()),		// disable driving
				new LambdaCommand(()->VisionServer.setStatistics(true)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(true)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_hub_pipeline))
			)
		).whileActiveOnce(
			new ParallelCommandGroup(
				this.cargo_sys.visionShoot(							// control the shooter with velocity determined by vision
					()->Xbox.Digital.A.getValueOf(this.input),		// press 'A' to feed
					Constants.feed_speed,
					(double inches)-> inches / 200.0 * 12.0			// 200 inches @ max power, 12v max voltage (obviously needs to be tuned)
				),
				new SequentialCommandGroup(							// LT and RT control turning speed of aim assist
					new HubFind.TeleopAssist(this.drivebase, ()->Xbox.Analog.RT.getValueOf(this.input) - Xbox.Analog.LT.getValueOf(this.input)),
					new HubTurn.TeleopAssist(this.drivebase, ()->Xbox.Analog.RT.getValueOf(this.input) - Xbox.Analog.LT.getValueOf(this.input))
				)
			)
		).whenInactive(
			new SequentialCommandGroup(
				new LambdaCommand(()->System.out.println("VISION ASSIST TERMINATED.")),
				new LambdaCommand(()->this.drivebase.modeDrive().schedule()),		// re-enable driving
				new LambdaCommand(()->VisionServer.setStatistics(false)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(false)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_driving))
			)
		);

	}
	private void arcadeControls() {	// bindings for arcade board

		TeleopTrigger.Get().whenActive(
			new SequentialCommandGroup(
				new LambdaCommand(()->VisionServer.setStatistics(false)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(false)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_driving))
			)
		).whenActive(
			this.drivebase.modeDrive(
				Attack3.Analog.X.getSupplier(this.stick_left),
				Attack3.Analog.Y.getSupplier(this.stick_left),
				Attack3.Analog.X.getSupplier(this.stick_right),
				Attack3.Analog.Y.getSupplier(this.stick_right),
				Attack3.Digital.TR.getPressedSupplier(this.stick_left),
				Attack3.Digital.TL.getPressedSupplier(this.stick_left)
			), false
		);	// schedule mode drive when in teleop mode

		//Attack3.Digital.TT.getCallbackFrom(this.stick_left).whenPressed(VisionSubsystem.IncrementPipeline.Get());
		//Attack3.Digital.TB.getCallbackFrom(this.stick_left).whenPressed(VisionSubsystem.DecrementPipeline.Get());
		Attack3.Digital.TT.getCallbackFrom(this.stick_left).whenPressed(VisionSubsystem.IncrementCamera.Get());
		Attack3.Digital.TB.getCallbackFrom(this.stick_left).whenPressed(VisionSubsystem.DecrementCamera.Get());
		//Attack3.Digital.TB.getCallbackFrom(this.stick_right).whenPressed(VisionSubsystem.ToggleProcessing.Get());
		//Attack3.Digital.TB.getCallbackFrom(this.stick_right).whenPressed(VisionSubsystem.ToggleStatistics.Get());

		Attack3.Digital.TL.getCallbackFrom(this.stick_right).and(
			Attack3.Digital.TRI.getCallbackFrom(this.stick_left).negate()
		).and( TeleopTrigger.Get() ).whileActiveOnce(
			this.cargo_sys.managedIntake(Constants.intake_speed)
		);
		Attack3.Digital.TL.getCallbackFrom(this.stick_right).and(
			Attack3.Digital.TRI.getCallbackFrom(this.stick_left)
		).and( TeleopTrigger.Get() ).whileActiveOnce(
			this.cargo_sys.basicIntake(Constants.intake_speed)
		);

		Attack3.Digital.TT.getCallbackFrom(this.stick_right).and(
			Attack3.Digital.TRI.getCallbackFrom(this.stick_left)
		).and( TeleopTrigger.Get() ).whileActiveOnce(
			this.cargo_sys.basicTransfer(Constants.transfer_speed)
		);
		Attack3.Digital.TR.getCallbackFrom(this.stick_right).and(
			Attack3.Digital.TRI.getCallbackFrom(this.stick_left).negate()
		).and(
			Attack3.Digital.TRI.getToggleFrom(this.stick_right).negate()
		).and( TeleopTrigger.Get() ).toggleWhenActive(
			this.cargo_sys.managedShoot(
				()->Attack3.Digital.TB.getValueOf(this.stick_right),
				Constants.feed_speed,
				Constants.shooter_default_speed
			)
		);
		Attack3.Digital.TR.getCallbackFrom(this.stick_right).and(
			Attack3.Digital.TRI.getCallbackFrom(this.stick_left)
		).and(
			Attack3.Digital.TRI.getToggleFrom(this.stick_right).negate()
		).and( TeleopTrigger.Get() ).toggleWhenActive(
			this.cargo_sys.basicShoot(
				()->Attack3.Digital.TB.getValueOf(this.stick_right),
				Constants.feed_speed,
				Constants.shooter_default_speed
			)
		);

		Attack3.Digital.TRI.getToggleFrom(this.stick_right).and(
			TeleopTrigger.Get()
		).whenActive(
			new SequentialCommandGroup(
				new LambdaCommand(()->System.out.println("VISION ASSIST RUNNING...")),
				new LambdaCommand(()->this.drivebase.modeDrive().cancel()),
				new LambdaCommand(()->VisionServer.setStatistics(true)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(true)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_hub_pipeline))
			)
		).whileActiveOnce(
			new ParallelCommandGroup(
				this.cargo_sys.visionShoot(
					()->Attack3.Digital.TB.getValueOf(this.stick_right),
					Constants.feed_speed,
					(double inches)-> inches / 200.0 * 12.0			// 200 inches @ max power, 12v max voltage (obviously needs to be tuned)
				),
				new SequentialCommandGroup(
					new HubFind.TeleopAssist(this.drivebase, Attack3.Analog.X.getSupplier(this.stick_left)),
					new HubTurn.TeleopAssist(this.drivebase, Attack3.Analog.X.getSupplier(this.stick_left))
				)
			)
		).whenInactive(
			new SequentialCommandGroup(
				new LambdaCommand(()->System.out.println("VISION ASSIST TERMINATED.")),
				new LambdaCommand(()->this.drivebase.modeDrive().schedule()),		// re-enable driving
				new LambdaCommand(()->VisionServer.setStatistics(false)),
				new LambdaCommand(()->VisionServer.setProcessingEnabled(false)),
				new LambdaCommand(()->VisionServer.applyCameraPreset(Constants.cam_driving))
			)
		);

	}


}